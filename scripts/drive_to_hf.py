"""Copies your AI models from Google Drive to Hugging Face (one private repo each).

Run it in Google Colab (Drive mounts there, and Colab's connection to Hugging Face
is fast, so a few GB takes minutes, not hours):

    from google.colab import drive, userdata
    drive.mount("/content/drive")
    import os; os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")      # a token with *write* access
    !pip install -q huggingface_hub
    !wget -q https://raw.githubusercontent.com/Tharunmaks/Mnx/claude/mnx-ai-web-interface-f75vu6/scripts/drive_to_hf.py
    !python drive_to_hf.py              # everything below
    !python drive_to_hf.py "Mnx 7B"     # or just one

What it does for each project:
  * big files split into parts (name.part00, name.part01, … or name_parts/ folders)
    are joined back into the real file first,
  * the folder is uploaded to  <your-username>/<repo name>  (private; use --public to change),
  * uploads can be resumed: just run it again.
Datasets go to dataset repos, models to model repos.
"""
import argparse
import os
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

# Drive folder (inside "My Drive") -> (Hugging Face repo name, repo type)
PROJECTS = {
    "Mnx 1.5 Max Coder": ("mnx-1.5-max-coder", "model"),
    "Mnx 5B MoE": ("mnx-5b-moe", "model"),
    "Mnx 1 7B model": ("mnx-1-7b", "model"),
    "Mnx 7B": ("mnx-7b", "model"),
    "Mnx Lite 1": ("mnx-lite-1", "model"),
    "Mnx lonjer": ("mnx-lonjer", "model"),
    "Mnx shared data": ("mnx-shared-data", "dataset"),
}
# Loose files in the top of My Drive that belong to a project: (name pattern) -> project folder name
LOOSE = {r"^es-(phone-bench|fused-addon)": ("mnx-es-experiments", "model")}

PART = re.compile(r"^(?P<base>.+)\.part(?P<n>\d+)$")


def join_parts(src: Path, dst: Path, log=print):
    """Copy src to dst, joining name.partNN files (and name_parts/ folders) into name.
    Files are streamed, so memory use stays small."""
    dst.mkdir(parents=True, exist_ok=True)
    groups = defaultdict(list)
    for item in sorted(src.iterdir()):
        if item.is_dir():
            if item.name.endswith("_parts"):
                join_parts(item, dst, log)  # parts folder: its joined file goes next to it
            else:
                join_parts(item, dst / item.name, log)
        else:
            m = PART.match(item.name)
            if m:
                groups[m["base"]].append((int(m["n"]), item))
            else:
                target = dst / item.name
                if not target.exists() or target.stat().st_size != item.stat().st_size:
                    shutil.copyfile(item, target)
    for base, parts in groups.items():
        parts.sort()
        nums = [n for n, _ in parts]
        if nums != list(range(nums[0], nums[0] + len(nums))):
            raise SystemExit(f"{src}/{base}: parts are missing (found {nums})")
        target = dst / base
        log(f"  joining {len(parts)} parts -> {base}")
        with open(target, "wb") as out:
            for _, p in parts:
                with open(p, "rb") as f:
                    shutil.copyfileobj(f, out, 16 * 1024 * 1024)


def upload(folder: Path, repo_id: str, repo_type: str, private: bool, dry_run: bool):
    if dry_run:
        size = sum(f.stat().st_size for f in folder.rglob("*") if f.is_file())
        print(f"  [dry run] would upload {size / 1e9:.2f} GB to {repo_type} {repo_id}")
        return
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(repo_id, repo_type=repo_type, private=private, exist_ok=True)
    api.upload_large_folder(repo_id=repo_id, repo_type=repo_type, folder_path=str(folder))
    print(f"  done: https://huggingface.co/{'datasets/' if repo_type == 'dataset' else ''}{repo_id}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("only", nargs="*", help="project names to copy (default: all)")
    ap.add_argument("--drive", default="/content/drive/MyDrive", help="where My Drive is mounted")
    ap.add_argument("--stage", default="/content/hf_stage", help="temporary folder for joined files")
    ap.add_argument("--public", action="store_true", help="make the repos public (default: private)")
    ap.add_argument("--dry-run", action="store_true", help="join and count, but upload nothing")
    ap.add_argument("--user", help="Hugging Face username (default: from HF_TOKEN)")
    a = ap.parse_args()

    root = Path(a.drive)
    if not root.is_dir():
        sys.exit(f"{root} not found. In Colab run: from google.colab import drive; drive.mount('/content/drive')")
    if not a.dry_run and not os.environ.get("HF_TOKEN"):
        sys.exit("Set HF_TOKEN (a Hugging Face token with write access): https://huggingface.co/settings/tokens")
    user = a.user
    if not user and not a.dry_run:
        from huggingface_hub import HfApi
        user = HfApi(token=os.environ["HF_TOKEN"]).whoami()["name"]
    user = user or "you"

    jobs = []
    for name, (repo, kind) in PROJECTS.items():
        if (not a.only or name in a.only) and (root / name).is_dir():
            jobs.append((name, root / name, repo, kind))
        elif a.only and name in a.only:
            print(f"! '{name}' isn't in {root}")
    for pattern, (repo, kind) in LOOSE.items():
        files = [f for f in root.iterdir() if f.is_file() and re.match(pattern, f.name)]
        if files and (not a.only or repo in a.only):
            jobs.append((repo, files, repo, kind))
    if not jobs:
        sys.exit("Nothing to copy.")

    for name, src, repo, kind in jobs:
        print(f"\n== {name}  ->  {user}/{repo} ({kind}, {'public' if a.public else 'private'})")
        stage = Path(a.stage) / repo
        if stage.exists():
            shutil.rmtree(stage)
        if isinstance(src, list):  # loose files
            tmp = Path(a.stage) / f"{repo}_src"
            tmp.mkdir(parents=True, exist_ok=True)
            for f in src:
                shutil.copyfile(f, tmp / f.name)
            join_parts(tmp, stage)
            shutil.rmtree(tmp)
        else:
            join_parts(src, stage)
        upload(stage, f"{user}/{repo}", kind, not a.public, a.dry_run)
        shutil.rmtree(stage, ignore_errors=True)  # free the disk for the next one
    print("\nAll done.")


if __name__ == "__main__":
    main()
