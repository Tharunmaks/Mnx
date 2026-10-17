// Mnx Studio — a Blender-style editor for the Bees' 3D models.
// The model is a list of parts (the same JSON the Bees write, hive/models3d.py); saving writes it and its .glb
// into the Bee's computer, ~/models/<name>.json + .glb.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { TransformControls } from "three/addons/controls/TransformControls.js";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";
import { api, fillIcons, gatewayBase, getSettings, h, installTokenPrompt } from "./shared.js";

const SHAPES = ["box", "sphere", "cylinder", "cone", "torus", "capsule", "plane"];
const NATURAL = { capsule: [0.5, 1, 0.5], torus: [1, 0.2, 1] }; // the unit shape's own size (as in models3d.py)
const $ = (id) => document.getElementById(id);
const params = new URLSearchParams(location.search);

let spec = { name: "new model", parts: [] };
let sel = -1;
let undo = [], redo = [];
let snap = false;

// ---------- the 3D view ----------
const canvas = $("canvas");
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.toneMapping = THREE.NeutralToneMapping;
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x1b1d24);
scene.environment = new THREE.PMREMGenerator(renderer).fromScene(new RoomEnvironment(), 0.04).texture;
scene.environmentIntensity = 0.9;
const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 5000);
camera.position.set(6, 4.5, 7);
const orbit = new OrbitControls(camera, canvas);
orbit.enableDamping = true;
orbit.target.set(0, 0.8, 0);
scene.add(new THREE.HemisphereLight(0xdfe8ff, 0x30281e, 0.5));
const sun = new THREE.DirectionalLight(0xffffff, 2.2);
sun.position.set(8, 14, 10);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
Object.assign(sun.shadow.camera, { left: -15, right: 15, top: 15, bottom: -15 });
scene.add(sun);
scene.add(new THREE.GridHelper(40, 40, 0x4a4f5c, 0x2c3039));
const axes = new THREE.AxesHelper(1.2);
axes.position.y = 0.002;
scene.add(axes);
const group = new THREE.Group();
scene.add(group);

const gizmo = new TransformControls(camera, canvas);
gizmo.addEventListener("dragging-changed", (e) => {
  orbit.enabled = !e.value;
  if (e.value) remember();
  else { readBack(); refreshProps(); }
});
gizmo.addEventListener("objectChange", () => readBack());
scene.add(gizmo.getHelper ? gizmo.getHelper() : gizmo);

function geometry(shape) {
  switch (shape) {
    case "sphere": return new THREE.SphereGeometry(0.5, 32, 16);
    case "cylinder": return new THREE.CylinderGeometry(0.5, 0.5, 1, 32);
    case "cone": return new THREE.ConeGeometry(0.5, 1, 32);
    case "torus": return new THREE.TorusGeometry(0.4, 0.1, 12, 48).rotateX(Math.PI / 2);
    case "capsule": return new THREE.CapsuleGeometry(0.25, 0.5, 8, 24);
    case "plane": return new THREE.PlaneGeometry(1, 1).rotateX(-Math.PI / 2);
    default: return new THREE.BoxGeometry(1, 1, 1);
  }
}
const GEOS = Object.fromEntries(SHAPES.map((s) => [s, geometry(s)]));

function meshFor(p, i) {
  const glow = p.glow || 0, opacity = p.opacity ?? 1;
  const mat = new THREE.MeshStandardMaterial({
    color: p.color, metalness: p.metal ?? 0, roughness: p.rough ?? 0.6,
    emissive: glow ? new THREE.Color(p.color) : 0x000000, emissiveIntensity: glow,
    transparent: opacity < 1, opacity, side: p.shape === "plane" ? THREE.DoubleSide : THREE.FrontSide,
  });
  const m = new THREE.Mesh(GEOS[p.shape] || GEOS.box, mat);
  const nat = NATURAL[p.shape] || [1, 1, 1];
  m.position.fromArray(p.pos);
  m.rotation.set(...p.rot.map(THREE.MathUtils.degToRad), "XYZ");
  m.scale.set(p.size[0] / nat[0], p.shape === "plane" ? 1 : p.size[1] / nat[1], p.size[2] / nat[2]);
  m.castShadow = m.receiveShadow = true;
  m.userData.i = i;
  return m;
}

const round = (x, n = 3) => Math.round(x * 10 ** n) / 10 ** n;

function readBack() { // the gizmo moved the mesh: write it into the part
  const m = gizmo.object;
  if (!m) return;
  const p = spec.parts[m.userData.i], nat = NATURAL[p.shape] || [1, 1, 1];
  p.pos = m.position.toArray().map((x) => round(x));
  p.rot = [m.rotation.x, m.rotation.y, m.rotation.z].map((r) => round(THREE.MathUtils.radToDeg(r), 2));
  p.size = [m.scale.x * nat[0], p.shape === "plane" ? p.size[1] : m.scale.y * nat[1], m.scale.z * nat[2]].map((x) => round(Math.abs(x)));
}

function rebuild() {
  gizmo.detach();
  for (const c of [...group.children]) { group.remove(c); c.material.dispose(); }
  spec.parts.forEach((p, i) => group.add(meshFor(p, i)));
  if (sel >= spec.parts.length) sel = -1;
  if (sel >= 0) gizmo.attach(group.children[sel]);
  outliner();
  refreshProps();
}

function select(i) {
  sel = i;
  if (i >= 0) gizmo.attach(group.children[i]); else gizmo.detach();
  outliner();
  refreshProps();
}

// ---------- undo ----------
function remember() {
  undo.push(JSON.stringify(spec));
  if (undo.length > 100) undo.shift();
  redo = [];
}
function step(from, to) {
  if (!from.length) return;
  to.push(JSON.stringify(spec));
  spec = JSON.parse(from.pop());
  rebuild();
}

// ---------- panels ----------
function outliner() {
  const ul = $("outliner");
  ul.replaceChildren(...spec.parts.map((p, i) => {
    const li = h("li", { class: i === sel ? "sel" : "" }, h("i", { style: `background:${p.color}` }), p.name);
    li.addEventListener("click", () => select(i));
    return li;
  }));
  $("count").textContent = spec.parts.length ? `(${spec.parts.length})` : "";
}

function refreshProps() {
  const p = spec.parts[sel];
  $("noSel").classList.toggle("hidden", !!p);
  $("propForm").classList.toggle("hidden", !p);
  if (!p) return;
  $("p-name").value = p.name;
  $("p-shape").value = p.shape;
  for (const inp of document.querySelectorAll(".vec input")) inp.value = p[inp.dataset.k][+inp.dataset.i];
  $("p-color").value = p.color;
  $("p-metal").value = p.metal ?? 0;
  $("p-rough").value = p.rough ?? 0.6;
  $("p-glow").value = p.glow ?? 0;
  $("p-opacity").value = p.opacity ?? 1;
}

function edit(fn) { // a change from the panel: remembered, applied, redrawn
  const p = spec.parts[sel];
  if (!p) return;
  remember();
  fn(p);
  const keep = sel;
  rebuild();
  select(keep);
}

$("p-shape").append(...SHAPES.map((s) => h("option", { value: s }, s)));
$("p-name").addEventListener("change", (e) => edit((p) => { p.name = e.target.value.slice(0, 60) || p.shape; }));
$("p-shape").addEventListener("change", (e) => edit((p) => { p.shape = e.target.value; }));
for (const inp of document.querySelectorAll(".vec input")) {
  inp.addEventListener("change", () => edit((p) => { const v = parseFloat(inp.value); if (Number.isFinite(v)) p[inp.dataset.k][+inp.dataset.i] = v; }));
}
$("p-color").addEventListener("input", (e) => edit((p) => { p.color = e.target.value; }));
for (const k of ["metal", "rough", "glow", "opacity"]) {
  $(`p-${k}`).addEventListener("change", (e) => edit((p) => { p[k] = parseFloat(e.target.value); }));
}

function add(shape) {
  remember();
  const size = shape === "plane" ? [4, 1, 4] : shape === "capsule" ? [0.5, 1, 0.5] : shape === "torus" ? [1, 0.2, 1] : [1, 1, 1];
  const y = shape === "plane" ? 0 : size[1] / 2;
  const t = orbit.target;
  spec.parts.push({ name: `${shape} ${spec.parts.length + 1}`, shape, size, pos: [round(t.x, 1), y, round(t.z, 1)], rot: [0, 0, 0],
    color: ["#d1202a", "#3a86ff", "#ffbe0b", "#8ac926", "#c9ced6", "#ff006e"][spec.parts.length % 6], rough: 0.6 });
  rebuild();
  select(spec.parts.length - 1);
}
$("adds").append(...SHAPES.map((s) => { const b = h("button", { type: "button" }, s); b.addEventListener("click", () => add(s)); return b; }));

function duplicate() {
  const p = spec.parts[sel];
  if (!p) return;
  remember();
  const copy = JSON.parse(JSON.stringify(p));
  copy.name = `${p.name} copy`;
  copy.pos[0] = round(copy.pos[0] + 0.5);
  spec.parts.push(copy);
  rebuild();
  select(spec.parts.length - 1);
}
function remove() {
  if (sel < 0) return;
  remember();
  spec.parts.splice(sel, 1);
  sel = -1;
  rebuild();
}
function focus() {
  const box = new THREE.Box3().setFromObject(sel >= 0 ? group.children[sel] : group);
  if (box.isEmpty()) return;
  const c = box.getCenter(new THREE.Vector3()), r = box.getSize(new THREE.Vector3()).length() || 1;
  orbit.target.copy(c);
  camera.position.copy(c).add(new THREE.Vector3(r * 0.9, r * 0.6, r * 1.1));
}
$("dup").addEventListener("click", duplicate);
$("del").addEventListener("click", remove);

function setMode(mode) {
  gizmo.setMode(mode);
  for (const b of document.querySelectorAll("#modes [data-mode]")) b.classList.toggle("on", b.dataset.mode === mode);
}
for (const b of document.querySelectorAll("#modes [data-mode]")) b.addEventListener("click", () => setMode(b.dataset.mode));
$("snap").addEventListener("click", () => {
  snap = !snap;
  $("snap").classList.toggle("on", snap);
  gizmo.setTranslationSnap(snap ? 0.1 : null);
  gizmo.setRotationSnap(snap ? THREE.MathUtils.degToRad(15) : null);
  gizmo.setScaleSnap(snap ? 0.1 : null);
});
$("undo").addEventListener("click", () => step(undo, redo));
$("redo").addEventListener("click", () => step(redo, undo));

addEventListener("keydown", (e) => {
  if (e.target.closest("input, select, textarea")) return;
  const k = e.key.toLowerCase();
  if ((e.ctrlKey || e.metaKey) && k === "s") { e.preventDefault(); save(); }
  else if ((e.ctrlKey || e.metaKey) && k === "z") { e.preventDefault(); e.shiftKey ? step(redo, undo) : step(undo, redo); }
  else if (e.shiftKey && k === "d") duplicate();
  else if (k === "g") setMode("translate");
  else if (k === "r") setMode("rotate");
  else if (k === "s") setMode("scale");
  else if (k === "x" || k === "delete") remove();
  else if (k === "f") focus();
  else if (k === "escape") select(-1);
});

// click a part to select it (a click, not an orbit drag)
const ray = new THREE.Raycaster();
let down = null;
canvas.addEventListener("pointerdown", (e) => { down = [e.clientX, e.clientY]; });
canvas.addEventListener("pointerup", (e) => {
  if (!down || Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 5 || gizmo.dragging) return;
  const r = canvas.getBoundingClientRect();
  ray.setFromCamera(new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1), camera);
  const hit = ray.intersectObjects(group.children, false)[0];
  select(hit ? hit.object.userData.i : -1);
});

function resize() {
  const r = canvas.parentElement.getBoundingClientRect();
  renderer.setSize(r.width, r.height, false);
  camera.aspect = r.width / Math.max(1, r.height);
  camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe(canvas.parentElement);
renderer.setAnimationLoop(() => { orbit.update(); renderer.render(scene, camera); });

// ---------- files in the Bee's computer ----------
let statusTimer;
function status(text) {
  const el = $("status");
  el.textContent = text;
  el.classList.add("show");
  clearTimeout(statusTimer);
  statusTimer = setTimeout(() => el.classList.remove("show"), 3500);
}
const bee = () => $("bee").value;
const slug = (s) => s.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 40);

async function listModels(pick) {
  if (!bee()) return;
  const { models } = await api(`/api/studio/${bee()}/models`);
  $("open").replaceChildren(h("option", { value: "" }, models.length ? "Open…" : "No models yet"), ...models.map((m) => h("option", { value: m }, m)));
  if (pick) $("open").value = pick;
}

async function openModel(name) {
  try {
    spec = await api(`/api/studio/${bee()}/models/${encodeURIComponent(name)}`);
    $("name").value = name;
    undo = []; redo = []; sel = -1;
    rebuild();
    focus();
    history.replaceState(null, "", `?bee=${bee()}&model=${name}`);
  } catch (err) { status(`Couldn't open it: ${err.message}`); }
}

async function save() {
  const name = slug($("name").value);
  if (!name) return status("Give it a name first");
  if (!spec.parts.length) return status("Add a part first");
  $("name").value = name;
  spec.name = name.replace(/-/g, " ");
  try {
    const r = await api(`/api/studio/${bee()}/models/${name}`, { method: "PUT", body: spec });
    status(`Saved to ${r.saved} (${Math.max(1, Math.round(r.bytes / 1024))} KB)`);
    await listModels(name);
    history.replaceState(null, "", `?bee=${bee()}&model=${name}`);
  } catch (err) { status(`Didn't save: ${err.message}`); }
}

$("save").addEventListener("click", save);
$("new").addEventListener("click", () => { remember(); spec = { name: "new model", parts: [] }; $("name").value = "new-model"; sel = -1; rebuild(); });
$("open").addEventListener("change", (e) => e.target.value && openModel(e.target.value));
$("bee").addEventListener("change", () => listModels());
$("view").addEventListener("click", async () => {
  await save();
  try {
    const r = await api(`/api/studio/${bee()}/models/${slug($("name").value)}/view`, { method: "POST" });
    open(r.url, "_blank", "noopener");
    status("Private viewer link opened (also in your Sandbox)");
  } catch (err) { status(err.message); }
});
$("glb").addEventListener("click", async () => {
  await save();
  const name = slug($("name").value);
  const res = await fetch(`${gatewayBase()}/api/studio/${bee()}/models/${name}/glb`, { headers: { Authorization: `Bearer ${getSettings().token}` } });
  if (!res.ok) return status("Couldn't make the .glb");
  const a = h("a", { href: URL.createObjectURL(await res.blob()), download: `${name}.glb` });
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 5000);
});

$("ask").addEventListener("submit", async (e) => {
  e.preventDefault();
  const text = $("askText").value.trim();
  if (!text || !bee()) return;
  const btn = e.target.querySelector("button");
  btn.disabled = true;
  btn.textContent = "Working…";
  try {
    const next = await api(`/api/studio/${bee()}/ask`, { method: "POST", body: { text, spec: spec.parts.length ? spec : null } });
    remember();
    const before = spec.parts.length;
    spec = next;
    sel = -1;
    rebuild();
    focus();
    $("askText").value = "";
    if (!before && $("name").value === "new-model") $("name").value = slug(next.name) || "new-model";
    status(`Done: ${next.parts.length} parts (Ctrl+Z to undo). Save to keep it.`);
  } catch (err) { status(err.message); }
  finally { btn.disabled = false; btn.textContent = "Do it"; }
});

async function start() {
  fillIcons();
  try {
    const { bees } = await api("/api/studio/bees");
    $("bee").replaceChildren(...(bees.length ? bees : [{ id: "", name: "No Bees yet" }]).map((b) => h("option", { value: b.id }, b.name)));
    if (params.get("bee") && bees.some((b) => b.id === params.get("bee"))) $("bee").value = params.get("bee");
    await listModels(params.get("model"));
    if (params.get("model")) await openModel(params.get("model"));
    else { add("box"); undo = []; }
  } catch (err) { status(err.message); }
}
installTokenPrompt(start);
start();
