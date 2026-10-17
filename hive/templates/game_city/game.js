// Neon Bay: a small open-world city. Drive, follow the traffic, run delivery missions, earn money.
// three.js from /vendor/three (served by the Hive's Chroma). Every part is plain code to change.
import * as THREE from "three";

const CITY = 12;          // blocks per side
const BLOCK = 40;         // block size (buildings + sidewalk)
const ROAD = 14;          // road width
const CELL = BLOCK + ROAD;
const HALF = (CITY * CELL) / 2;

// ---------- renderer, scene, camera ----------
const canvas = document.getElementById("game");
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(65, 1, 0.5, 2000);
const DAY = { sky: 0x8fc8ff, fog: 0x9fd0ff, sun: 1.6, hemi: 0.9 };
const NIGHT = { sky: 0x070a1c, fog: 0x0a0d24, sun: 0.15, hemi: 0.25 };
scene.fog = new THREE.Fog(DAY.fog, 120, 520);
scene.background = new THREE.Color(DAY.sky);
const hemi = new THREE.HemisphereLight(0xffffff, 0x334455, DAY.hemi);
scene.add(hemi);
const sun = new THREE.DirectionalLight(0xffffff, DAY.sun);
sun.position.set(120, 220, 80);
sun.castShadow = true;
sun.shadow.mapSize.set(1024, 1024);
Object.assign(sun.shadow.camera, { left: -160, right: 160, top: 160, bottom: -160, far: 600 });
scene.add(sun, sun.target);

function resize() {
  const w = innerWidth, h = innerHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
addEventListener("resize", resize);
resize();

// ---------- the city ----------
const ground = new THREE.Mesh(new THREE.PlaneGeometry(CITY * CELL + 400, CITY * CELL + 400), new THREE.MeshLambertMaterial({ color: 0x2b2f38 }));
ground.rotation.x = -Math.PI / 2;
ground.receiveShadow = true;
scene.add(ground);

const rand = (() => { let s = 7; return () => (s = (s * 16807) % 2147483647) / 2147483647; })();  // same city every time
const solids = [];  // building footprints the car can't drive through: {x0, x1, z0, z1}
const boxGeo = new THREE.BoxGeometry(1, 1, 1);
const palette = [0x5b6b8c, 0x8c6b5b, 0x6b8c7a, 0xb0a48a, 0x7a6b8c, 0x9aa3ad, 0x48566b];
const windowsMat = new THREE.MeshLambertMaterial({ color: 0xffe9a8, emissive: 0x000000 });
const buildingCount = CITY * CITY * 4;
const buildings = new THREE.InstancedMesh(boxGeo, new THREE.MeshLambertMaterial(), buildingCount);
buildings.castShadow = buildings.receiveShadow = true;
const glow = new THREE.InstancedMesh(boxGeo, windowsMat, buildingCount);
const m = new THREE.Matrix4(), q = new THREE.Quaternion(), color = new THREE.Color();
let n = 0;
for (let i = 0; i < CITY; i++) {
  for (let j = 0; j < CITY; j++) {
    const cx = -HALF + i * CELL + ROAD + BLOCK / 2, cz = -HALF + j * CELL + ROAD + BLOCK / 2;
    const park = rand() < 0.1;
    const walk = new THREE.Mesh(new THREE.BoxGeometry(BLOCK, 0.4, BLOCK), new THREE.MeshLambertMaterial({ color: park ? 0x3f7a3a : 0x6d717a }));
    walk.position.set(cx, 0.2, cz);
    walk.receiveShadow = true;
    scene.add(walk);
    if (park) { addTrees(cx, cz); continue; }
    for (const [ox, oz] of [[-1, -1], [1, -1], [-1, 1], [1, 1]]) {
      const w = 12 + rand() * 6, d = 12 + rand() * 6, h = 10 + rand() ** 2 * (Math.hypot(cx, cz) < 150 ? 120 : 45);
      const x = cx + ox * BLOCK / 4, z = cz + oz * BLOCK / 4;
      m.compose(new THREE.Vector3(x, h / 2 + 0.4, z), q, new THREE.Vector3(w, h, d));
      buildings.setMatrixAt(n, m);
      buildings.setColorAt(n, color.setHex(palette[Math.floor(rand() * palette.length)]));
      m.compose(new THREE.Vector3(x, h / 2 + 0.4, z), q, new THREE.Vector3(w + 0.1, h * 0.92, d + 0.1));
      glow.setMatrixAt(n, m);
      solids.push({ x0: x - w / 2 - 1, x1: x + w / 2 + 1, z0: z - d / 2 - 1, z1: z + d / 2 + 1 });
      n++;
    }
  }
}
buildings.count = glow.count = n;
glow.visible = false;
scene.add(buildings, glow);

function addTrees(cx, cz) {
  for (let k = 0; k < 6; k++) {
    const x = cx + (rand() - 0.5) * BLOCK * 0.8, z = cz + (rand() - 0.5) * BLOCK * 0.8;
    const trunk = new THREE.Mesh(new THREE.CylinderGeometry(0.4, 0.5, 3), new THREE.MeshLambertMaterial({ color: 0x6b4a2b }));
    const top = new THREE.Mesh(new THREE.SphereGeometry(2.6, 8, 6), new THREE.MeshLambertMaterial({ color: 0x2f8f3a }));
    trunk.position.set(x, 1.9, z);
    top.position.set(x, 5, z);
    top.castShadow = true;
    scene.add(trunk, top);
  }
}

// road markings
const lineMat = new THREE.MeshBasicMaterial({ color: 0xe8e8e8 });
for (let i = 0; i <= CITY; i++) {
  const p = -HALF + i * CELL + ROAD / 2;
  for (const horiz of [true, false]) {
    const line = new THREE.Mesh(new THREE.PlaneGeometry(horiz ? CITY * CELL : 0.4, horiz ? 0.4 : CITY * CELL), lineMat);
    line.rotation.x = -Math.PI / 2;
    line.position.set(horiz ? 0 : p, 0.05, horiz ? p : 0);
    scene.add(line);
  }
}

// ---------- cars ----------
function makeCar(body) {
  const car = new THREE.Group();
  const paint = new THREE.MeshLambertMaterial({ color: body });
  const base = new THREE.Mesh(new THREE.BoxGeometry(2.2, 0.9, 4.4), paint);
  base.position.y = 0.9;
  const cabin = new THREE.Mesh(new THREE.BoxGeometry(1.9, 0.8, 2.2), new THREE.MeshLambertMaterial({ color: 0x1b2633 }));
  cabin.position.set(0, 1.75, -0.2);
  car.add(base, cabin);
  for (const [x, z] of [[-1.1, 1.4], [1.1, 1.4], [-1.1, -1.4], [1.1, -1.4]]) {
    const wheel = new THREE.Mesh(new THREE.CylinderGeometry(0.45, 0.45, 0.4, 12), new THREE.MeshLambertMaterial({ color: 0x111111 }));
    wheel.rotation.z = Math.PI / 2;
    wheel.position.set(x, 0.45, z);
    car.add(wheel);
  }
  const light = new THREE.Mesh(new THREE.BoxGeometry(1.8, 0.2, 0.1), new THREE.MeshBasicMaterial({ color: 0xfff6c0 }));
  light.position.set(0, 1, 2.21);
  car.add(light);
  car.traverse((o) => { o.castShadow = true; });
  return car;
}
const player = makeCar(0xff2d8a);
scene.add(player);
const state = { speed: 0, heading: 0, money: 0, night: false, cam: 0 };
function respawn() {
  player.position.set(-HALF + ROAD / 2, 0, -HALF + CELL * 3);
  state.speed = 0;
  state.heading = 0;
}
respawn();

// traffic: cars driving along the roads, turning at crossings
const traffic = [];
for (let k = 0; k < 40; k++) {
  const c = makeCar(palette[k % palette.length] + 0x202020);
  const lane = Math.floor(rand() * (CITY + 1)), alongX = rand() < 0.5;
  const p = -HALF + lane * CELL + ROAD / 2 + (rand() < 0.5 ? -3 : 3);
  c.position.set(alongX ? (rand() - 0.5) * CITY * CELL : p, 0, alongX ? p : (rand() - 0.5) * CITY * CELL);
  c.userData = { dir: alongX ? new THREE.Vector3(rand() < 0.5 ? 1 : -1, 0, 0) : new THREE.Vector3(0, 0, rand() < 0.5 ? 1 : -1), v: 10 + rand() * 8 };
  c.rotation.y = Math.atan2(c.userData.dir.x, c.userData.dir.z);
  scene.add(c);
  traffic.push(c);
}

// ---------- missions: pick up at the marker, deliver to the next ----------
const marker = new THREE.Mesh(new THREE.CylinderGeometry(4, 4, 30, 24, 1, true),
  new THREE.MeshBasicMaterial({ color: 0xffe066, transparent: true, opacity: 0.35, side: THREE.DoubleSide }));
scene.add(marker);
const missions = ["Pick up the package", "Deliver it to the docks", "Pick up a passenger", "Drop them downtown", "Grab the parts", "Bring them to the garage"];
let mission = 0;
function placeMarker() {
  const i = Math.floor(rand() * CITY), j = Math.floor(rand() * CITY);
  marker.position.set(-HALF + i * CELL + ROAD / 2, 15, -HALF + j * CELL + CELL / 2);
  document.getElementById("mission").textContent = `${missions[mission % missions.length]} (yellow marker)`;
}
placeMarker();

function toast(text) {
  const t = document.getElementById("toast");
  t.textContent = text;
  t.style.opacity = 1;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => { t.style.opacity = 0; }, 1600);
}

// ---------- controls: keyboard and touch ----------
const keys = new Set();
addEventListener("keydown", (e) => {
  keys.add(e.key.toLowerCase());
  if (e.key.toLowerCase() === "c") state.cam = (state.cam + 1) % 2;
  if (e.key.toLowerCase() === "n") setNight(!state.night);
  if (e.key.toLowerCase() === "r") respawn();
});
addEventListener("keyup", (e) => keys.delete(e.key.toLowerCase()));
for (const b of document.querySelectorAll("#touch button")) {
  const k = b.dataset.k;
  b.addEventListener("pointerdown", (e) => { e.preventDefault(); keys.add(k); });
  for (const ev of ["pointerup", "pointerleave", "pointercancel"]) b.addEventListener(ev, () => keys.delete(k));
}
const down = (...names) => names.some((k) => keys.has(k));

function setNight(on) {
  state.night = on;
  const s = on ? NIGHT : DAY;
  scene.background.setHex(s.sky);
  scene.fog.color.setHex(s.fog);
  sun.intensity = s.sun;
  hemi.intensity = s.hemi;
  glow.visible = on;
  windowsMat.emissive.setHex(on ? 0x9a7a2a : 0x000000);
}

// ---------- physics ----------
function hits(x, z) {
  if (Math.abs(x) > HALF + 60 || Math.abs(z) > HALF + 60) return true;
  return solids.some((s) => x > s.x0 && x < s.x1 && z > s.z0 && z < s.z1);
}

const clock = new THREE.Clock();
const camPos = new THREE.Vector3();
function frame() {
  const dt = Math.min(clock.getDelta(), 0.05);
  const gas = down("w", "arrowup", "gas"), back = down("s", "arrowdown"), brake = down(" ", "brake");
  const left = down("a", "arrowleft", "left"), right = down("d", "arrowright", "right");
  if (gas) state.speed += 28 * dt;
  if (back) state.speed -= 20 * dt;
  if (brake) state.speed *= 1 - 3 * dt;
  state.speed *= 1 - 0.6 * dt;
  state.speed = THREE.MathUtils.clamp(state.speed, -15, 55);
  const steer = (left ? 1 : 0) - (right ? 1 : 0);
  state.heading += steer * dt * 1.9 * Math.min(1, Math.abs(state.speed) / 8) * Math.sign(state.speed || 1);
  const nx = player.position.x + Math.sin(state.heading) * state.speed * dt;
  const nz = player.position.z + Math.cos(state.heading) * state.speed * dt;
  if (hits(nx, nz)) { state.speed *= -0.3; } else { player.position.x = nx; player.position.z = nz; }
  player.rotation.y = state.heading;

  for (const c of traffic) {
    const d = c.userData;
    c.position.addScaledVector(d.dir, d.v * dt);
    if (Math.abs(c.position.x) > HALF) c.position.x = -Math.sign(c.position.x) * HALF;
    if (Math.abs(c.position.z) > HALF) c.position.z = -Math.sign(c.position.z) * HALF;
    if (c.position.distanceTo(player.position) < 3.2) { state.speed *= -0.4; c.position.addScaledVector(d.dir, 2); }
  }

  marker.rotation.y += dt;
  if (Math.hypot(player.position.x - marker.position.x, player.position.z - marker.position.z) < 5) {
    const pay = 100 + Math.floor(rand() * 400);
    state.money += pay;
    document.getElementById("money").textContent = `$${state.money.toLocaleString()}`;
    toast(`${missions[mission % missions.length]}: done  +$${pay}`);
    mission++;
    placeMarker();
  }

  // chase camera (or a high view)
  const back3 = state.cam === 0 ? 11 : 26, up = state.cam === 0 ? 5 : 22;
  camPos.set(player.position.x - Math.sin(state.heading) * back3, up, player.position.z - Math.cos(state.heading) * back3);
  camera.position.lerp(camPos, 1 - Math.pow(0.001, dt));
  camera.lookAt(player.position.x, 2, player.position.z);
  sun.position.set(player.position.x + 120, 220, player.position.z + 80);
  sun.target.position.copy(player.position);

  document.getElementById("speed").textContent = `${Math.round(Math.abs(state.speed) * 3.6)} km/h`;
  drawMap();
  renderer.render(scene, camera);
  requestAnimationFrame(frame);
}

// ---------- minimap ----------
const map = document.getElementById("map").getContext("2d");
function drawMap() {
  const s = 140 / (CITY * CELL + 60);
  map.clearRect(0, 0, 140, 140);
  map.fillStyle = "rgba(255,255,255,.18)";
  for (let i = 0; i <= CITY; i++) {
    const p = (-HALF + i * CELL + ROAD / 2 + HALF + 30) * s;
    map.fillRect(p - 1, 0, 2, 140);
    map.fillRect(0, p - 1, 140, 2);
  }
  const dot = (x, z, c, r) => { map.fillStyle = c; map.beginPath(); map.arc((x + HALF + 30) * s, (z + HALF + 30) * s, r, 0, 7); map.fill(); };
  dot(marker.position.x, marker.position.z, "#ffe066", 5);
  dot(player.position.x, player.position.z, "#ff2d8a", 4);
}

window.__game = { state, player, ready: true };  // lets a Bee check that the game runs
frame();
