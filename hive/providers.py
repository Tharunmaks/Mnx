"""Cloud GPU providers the Cloud Bee can rent from. First one: RunPod.

Renting costs money, so every rent goes through a confirm card that shows the price,
and rented machines are listed with a Stop button. The API key is a Lab secret
(RUNPOD_API_KEY). Terminating a pod stops the billing.
"""

from __future__ import annotations

import asyncio
import re
import time

import httpx

RUNPOD_URL = "https://api.runpod.io/graphql"
# A PyTorch image RunPod publishes, with an SSH server that trusts PUBLIC_KEY.
RUNPOD_IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"


class ProviderError(RuntimeError):
    pass


async def _gql(key: str, query: str, variables: dict | None = None) -> dict:
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(RUNPOD_URL, params={"api_key": key}, json={"query": query, "variables": variables or {}})
    if r.status_code == 401:
        raise ProviderError("RunPod rejected the API key")
    try:
        data = r.json()
    except ValueError as exc:
        raise ProviderError(f"RunPod answered {r.status_code}: {r.text[:200]}") from exc
    if data.get("errors"):
        raise ProviderError("RunPod: " + "; ".join(e.get("message", "?") for e in data["errors"])[:300])
    return data.get("data") or {}


async def runpod_gpu_types(key: str) -> list[dict]:
    """GPU types with memory and the current on-demand price per hour."""
    data = await _gql(key, """
      query { gpuTypes { id displayName memoryInGb secureCloud communityCloud
        lowestPrice(input: {gpuCount: 1}) { uninterruptablePrice minimumBidPrice stockStatus } } }""")
    out = []
    for g in data.get("gpuTypes") or []:
        price = (g.get("lowestPrice") or {}).get("uninterruptablePrice")
        if not price or not g.get("memoryInGb"):
            continue
        out.append({"id": g["id"], "name": g.get("displayName") or g["id"], "memory_gb": g["memoryInGb"],
                    "price": round(float(price), 2), "stock": (g.get("lowestPrice") or {}).get("stockStatus")})
    out.sort(key=lambda x: x["price"])
    return out


async def runpod_rent(key: str, gpu_type: str, count: int, public_key: str, name: str = "mnx-hive",
                      disk_gb: int = 100) -> dict:
    """Start an on-demand pod with SSH. Returns {id, name}. Billing starts now."""
    if not re.fullmatch(r"[\w .+-]{1,80}", gpu_type):
        raise ProviderError("Bad GPU type")
    data = await _gql(key, """
      mutation Rent($input: PodFindAndDeployOnDemandInput) {
        podFindAndDeployOnDemand(input: $input) { id name } }""", {"input": {
        "cloudType": "ALL", "gpuCount": int(count), "gpuTypeId": gpu_type, "name": name,
        "imageName": RUNPOD_IMAGE, "volumeInGb": int(disk_gb), "containerDiskInGb": 50,
        "volumeMountPath": "/workspace", "ports": "22/tcp", "minVcpuCount": 4, "minMemoryInGb": 16,
        "env": [{"key": "PUBLIC_KEY", "value": public_key}],
    }})
    pod = data.get("podFindAndDeployOnDemand")
    if not pod:
        raise ProviderError("RunPod didn't start a pod (no stock for that GPU?)")
    return {"id": pod["id"], "name": pod.get("name") or name}


async def runpod_pod(key: str, pod_id: str) -> dict:
    data = await _gql(key, """
      query Pod($id: String!) { pod(input: {podId: $id}) { id name desiredStatus costPerHr
        runtime { uptimeInSeconds ports { ip isIpPublic privatePort publicPort type } gpus { id gpuUtilPercent memoryUtilPercent } }
        machine { gpuDisplayName } } }""", {"id": pod_id})
    pod = data.get("pod")
    if not pod:
        raise ProviderError("That pod no longer exists")
    ssh = None
    for p in (pod.get("runtime") or {}).get("ports") or []:
        if p.get("privatePort") == 22 and p.get("isIpPublic") and p.get("type") == "tcp":
            ssh = {"host": p["ip"], "port": p["publicPort"]}
    return {"id": pod["id"], "name": pod.get("name"), "status": pod.get("desiredStatus"), "cost_per_hour": pod.get("costPerHr"),
            "gpu": (pod.get("machine") or {}).get("gpuDisplayName"), "ssh": ssh,
            "uptime": (pod.get("runtime") or {}).get("uptimeInSeconds")}


async def runpod_wait_ssh(key: str, pod_id: str, timeout: float = 600) -> dict:
    """Poll until the pod has a public SSH port."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        pod = await runpod_pod(key, pod_id)
        if pod["ssh"]:
            return pod
        await asyncio.sleep(10)
    raise ProviderError("The pod started but SSH never came up (10 minutes); check it on runpod.io")


async def runpod_stop(key: str, pod_id: str) -> None:
    await _gql(key, "mutation Stop($id: String!) { podTerminate(input: {podId: $id}) }", {"id": pod_id})
