// Neon Bay: an open-world city. Drive, drift, run delivery missions, earn money. Day, night, rain.
// The picture comes from gfx.js (sky, light, shadows, bloom, cars, textures); this file is the city and the rules.
import * as THREE from "three";
import { mergeGeometries } from "three/addons/utils/BufferGeometryUtils.js";
import { createStage, buildCar, animateCar, facadeTextures, asphaltTextures, concreteTextures, makeWater, lightPoolTexture,
  Particles, makeAudio, rng } from "./gfx.js";

const CITY = 12;          // blocks per side
const BLOCK = 40;         // block size (buildings + sidewalk)
const ROAD = 14;          // road width
const CELL = BLOCK + ROAD;
const HALF = (CITY * CELL) / 2;
const EDGE = HALF + 90;   // the island's edge
const $ = (id) => document.getElementById(id);

const stage = createStage($("game"), { time: 17.3 });
const { scene, camera } = stage;
const rand = rng(7);

// ---------- materials
const asphalt = asphaltTextures(), concrete = concreteTextures();
asphalt.map.repeat.set((EDGE * 2) / 9, (EDGE * 2) / 9);
asphalt.rough.repeat.copy(asphalt.map.repeat);
const roadMat = stage.wet(new THREE.MeshStandardMaterial({ map: asphalt.map, roughnessMap: asphalt.rough, roughness: 1, metalness: 0.05 }), 0.22, 2.2);
const walkMat = stage.wet(new THREE.MeshStandardMaterial({ map: concrete.map, roughness: 0.85, color: 0xd5d7dd }), 0.4, 1.6);
const paintOn = { roughness: 0.6, polygonOffset: true, polygonOffsetFactor: -2, polygonOffsetUnits: -2 };
const markYellow = new THREE.MeshStandardMaterial({ color: 0xe9b92a, ...paintOn });
const markWhite = new THREE.MeshStandardMaterial({ color: 0xe8e8e8, ...paintOn });

// ---------- ground: the sea, the island, the roads
const water = makeWater(16000, -0.9);
scene.add(water);
const island = new THREE.Mesh(new THREE.BoxGeometry(EDGE * 2, 4, EDGE * 2), new THREE.MeshStandardMaterial({ color: 0x6f737b, roughness: 0.9 }));
island.position.y = -2.05; island.receiveShadow = true;
scene.add(island);
const ground = new THREE.Mesh(new THREE.PlaneGeometry(EDGE * 2, EDGE * 2), roadMat);
ground.rotation.x = -Math.PI / 2; ground.receiveShadow = true;
scene.add(ground);
const wallMat = new THREE.MeshStandardMaterial({ color: 0x4d5159, roughness: 0.8 });
for (const [x, z, w, d] of [[0, EDGE, EDGE * 2 + 2, 2], [0, -EDGE, EDGE * 2 + 2, 2], [EDGE, 0, 2, EDGE * 2], [-EDGE, 0, 2, EDGE * 2]]) {
  const wall = new THREE.Mesh(new THREE.BoxGeometry(w, 1.4, d), wallMat);
  wall.position.set(x, 0.5, z); wall.castShadow = wall.receiveShadow = true; scene.add(wall);
}

// ---------- the blocks: sidewalks, buildings, parks, trees, lamps, signs
const facades = [1, 2, 3, 4, 5, 6].map((s) => facadeTextures(s, 0.3 + s * 0.05));
const bld = facades.map(() => []);
const tints = [0xffffff, 0xf2e6d4, 0xd9e2ee, 0xe6d5c4, 0xcfd6dd, 0xdcc9b6, 0xb9c3d1].map((c) => new THREE.Color(c));
const solids = [], walkGeos = [], roofBits = [], lampPos = [], treePos = [], signs = [], parks = [];
const box = new THREE.BoxGeometry(1, 1, 1);

function building(x, z, w, d, h, type, y0 = 0) {
  const f = facades[type], tint = tints[Math.floor(rand() * tints.length)], offU = rand(), offV = rand();
  const g = new THREE.BoxGeometry(w, h, d);
  const uv = g.attributes.uv, col = new Float32Array(uv.count * 3);
  for (let face = 0; face < 6; face++) {
    for (let i = 0; i < 4; i++) {
      const k = face * 4 + i;
      if (face === 2 || face === 3) uv.setXY(k, 0.012, 0.012);                 // roof and floor: plain wall
      else uv.setXY(k, uv.getX(k) * (face < 2 ? d : w) / f.tile.x + offU, uv.getY(k) * h / f.tile.y + offV);
      col.set([tint.r, tint.g, tint.b], k * 3);
    }
  }
  g.setAttribute("color", new THREE.BufferAttribute(col, 3));
  g.translate(x, y0 + h / 2 + 0.3, z);
  bld[type].push(g);
}

for (let i = 0; i < CITY; i++) {
  for (let j = 0; j < CITY; j++) {
    const cx = -HALF + i * CELL + ROAD + BLOCK / 2, cz = -HALF + j * CELL + ROAD + BLOCK / 2;
    const park = rand() < 0.1;
    const slab = new THREE.BoxGeometry(BLOCK, 0.3, BLOCK);
    const uvs = slab.attributes.uv;
    for (let k = 0; k < uvs.count; k++) uvs.setXY(k, uvs.getX(k) * BLOCK / 5, uvs.getY(k) * BLOCK / 5);
    walkGeos.push(slab.translate(cx, 0.15, cz));
    if (park) { parks.push({ cx, cz }); continue; }
    const downtown = Math.hypot(cx, cz) < 170;
    for (const [ox, oz] of [[-1, -1], [1, -1], [-1, 1], [1, 1]]) {
      const w = 13 + rand() * 5, d = 13 + rand() * 5;
      const h = 9 + rand() ** 2 * (downtown ? 125 : 42) + (downtown ? 14 : 0);
      const x = cx + ox * BLOCK / 4, z = cz + oz * BLOCK / 4, type = Math.floor(rand() * facades.length);
      building(x, z, w, d, h, type);
      if (h > 55) building(x, z, w * 0.62, d * 0.62, h * 0.22 + 4, type, h);       // a setback tier on tall towers
      solids.push({ x0: x - w / 2 - 1.2, x1: x + w / 2 + 1.2, z0: z - d / 2 - 1.2, z1: z + d / 2 + 1.2 });
      roofBits.push({ x, z, y: (h > 55 ? h * 1.22 + 4 : h) + 0.3, w: h > 55 ? w * 0.62 : w, d: h > 55 ? d * 0.62 : d, tall: h > 40 });
      if (h > 14 && rand() < 0.55) {                                             // neon signs on the faces that look at the street
        const side = rand() < 0.5 ? "x" : "z";
        signs.push({ x: side === "x" ? x + ox * (w / 2 + 0.12) : x + (rand() - 0.5) * w * 0.5, z: side === "z" ? z + oz * (d / 2 + 0.12) : z + (rand() - 0.5) * d * 0.5,
          y: 5 + rand() * Math.min(h - 8, 22), ry: side === "x" ? (ox < 0 ? -Math.PI / 2 : Math.PI / 2) : (oz < 0 ? Math.PI : 0), kind: Math.floor(rand() * 6) });
      }
    }
    for (const [ax, az] of [[-1, -1], [1, -1], [-1, 1], [1, 1]]) lampPos.push([cx + ax * (BLOCK / 2 - 1.1), cz + az * (BLOCK / 2 - 1.1)]);
    if (stage.level > 0) for (let k = 0; k < 5; k++) {                           // street trees along the kerb
      const t = (k + 0.5) / 5 - 0.5;
      treePos.push([cx + t * (BLOCK - 8), cz + (BLOCK / 2 - 1.1) * (k % 2 ? 1 : -1), 0.8 + rand() * 0.5]);
    }
  }
}
facades.forEach((f, t) => {
  if (!bld[t].length) return;
  const mat = new THREE.MeshStandardMaterial({ map: f.map, emissiveMap: f.emissive, emissive: 0xffffff, roughnessMap: f.orm, metalnessMap: f.orm,
    roughness: 1, metalness: 1, vertexColors: true, envMapIntensity: 1.1 });
  stage.glow(mat, 0.0, 2.4);
  stage.wet(mat, 0.6, 1.3);
  const m = new THREE.Mesh(mergeGeometries(bld[t]), mat);
  m.castShadow = m.receiveShadow = true;
  scene.add(m);
});
const sidewalks = new THREE.Mesh(mergeGeometries(walkGeos), walkMat);
sidewalks.receiveShadow = true; sidewalks.castShadow = true;
scene.add(sidewalks);

// rooftops: air-conditioners and blinking aerials
const M = new THREE.Matrix4(), Q = new THREE.Quaternion(), P = new THREE.Vector3(), S = new THREE.Vector3(), E = new THREE.Euler();
const acs = new THREE.InstancedMesh(box, new THREE.MeshStandardMaterial({ color: 0x8c9099, roughness: 0.7, metalness: 0.3 }), roofBits.length * 3);
const aerials = new THREE.InstancedMesh(new THREE.CylinderGeometry(0.12, 0.2, 1, 6), new THREE.MeshStandardMaterial({ color: 0x3a3d44, roughness: 0.6, metalness: 0.6 }), roofBits.length);
const blinkMat = new THREE.MeshStandardMaterial({ color: 0x200000, emissive: 0xff2020, emissiveIntensity: 0 });
const blinks = new THREE.InstancedMesh(new THREE.SphereGeometry(0.45, 8, 6), blinkMat, roofBits.length);
let ai = 0, pi = 0;
for (const r of roofBits) {
  for (let k = 0; k < 3; k++) {
    S.set(1.6 + rand() * 2.6, 0.9 + rand() * 1.1, 1.4 + rand() * 2.2);
    P.set(r.x + (rand() - 0.5) * (r.w - 5), r.y + S.y / 2, r.z + (rand() - 0.5) * (r.d - 5));
    acs.setMatrixAt(ai++, M.compose(P, Q.setFromEuler(E.set(0, rand() * 3, 0)), S));
  }
  if (r.tall) {
    const hgt = 9 + rand() * 12;
    aerials.setMatrixAt(pi, M.compose(P.set(r.x, r.y + hgt / 2, r.z), Q.identity(), S.set(1, hgt, 1)));
    blinks.setMatrixAt(pi++, M.compose(P.set(r.x, r.y + hgt + 0.3, r.z), Q.identity(), S.set(1, 1, 1)));
  }
}
acs.count = ai; aerials.count = blinks.count = pi;
acs.castShadow = aerials.castShadow = true;
scene.add(acs, aerials, blinks);

// neon signs
const signTex = [["BAR", "#ff2d8a"], ["RAMEN", "#ffb400"], ["HOTEL", "#2de2ff"], ["24H", "#8aff3d"], ["CLUB", "#b36bff"], ["PIZZA", "#ff5a36"]].map(([text, color]) => {
  const c = document.createElement("canvas"); c.width = 256; c.height = 128;
  const g = c.getContext("2d");
  g.fillStyle = "#07080d"; g.fillRect(0, 0, 256, 128);
  g.strokeStyle = color; g.lineWidth = 7; g.shadowColor = color; g.shadowBlur = 14; g.strokeRect(10, 10, 236, 108);
  g.fillStyle = color; g.font = "800 62px Inter, Arial, sans-serif"; g.textAlign = "center"; g.textBaseline = "middle"; g.fillText(text, 128, 68);
  const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace; t.anisotropy = 4;
  return t;
});
signTex.forEach((tx, kind) => {
  const mine = signs.filter((s) => s.kind === kind);
  if (!mine.length) return;
  const mat = new THREE.MeshStandardMaterial({ map: tx, emissiveMap: tx, emissive: 0xffffff, roughness: 0.5 });
  stage.glow(mat, 0.35, 3);
  const im = new THREE.InstancedMesh(new THREE.PlaneGeometry(7, 3.5), mat, mine.length);
  mine.forEach((s, i) => im.setMatrixAt(i, M.compose(P.set(s.x, s.y, s.z), Q.setFromEuler(E.set(0, s.ry, 0)), S.set(1, 1, 1))));
  scene.add(im);
});

// street lamps, and the light they throw on the road at night
const lampPoles = new THREE.InstancedMesh(new THREE.CylinderGeometry(0.1, 0.16, 8, 8), new THREE.MeshStandardMaterial({ color: 0x2c2f36, roughness: 0.5, metalness: 0.7 }), lampPos.length);
const bulbMat = new THREE.MeshStandardMaterial({ color: 0xfff1d0, emissive: 0xffd9a0 });
stage.glow(bulbMat, 0.1, 5);
const bulbs = new THREE.InstancedMesh(new THREE.SphereGeometry(0.42, 10, 8), bulbMat, lampPos.length);
const poolMat = new THREE.MeshBasicMaterial({ map: lightPoolTexture(), transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, opacity: 0, polygonOffset: true, polygonOffsetFactor: -3 });
const pools = new THREE.InstancedMesh(new THREE.PlaneGeometry(1, 1).rotateX(-Math.PI / 2), poolMat, lampPos.length);
lampPos.forEach(([x, z], i) => {
  lampPoles.setMatrixAt(i, M.compose(P.set(x, 4.2, z), Q.identity(), S.set(1, 1, 1)));
  bulbs.setMatrixAt(i, M.compose(P.set(x, 8.3, z), Q.identity(), S.set(1, 0.7, 1)));
  pools.setMatrixAt(i, M.compose(P.set(x, 0.33, z), Q.identity(), S.set(15, 1, 15)));
});
lampPoles.castShadow = true;
scene.add(lampPoles, bulbs, pools);

// trees: a trunk and three clumps of leaves each
const treeCount = treePos.length + parks.length * 9;
const trunkI = new THREE.InstancedMesh(new THREE.CylinderGeometry(0.22, 0.34, 3.4, 7), new THREE.MeshStandardMaterial({ color: 0x5a3f27, roughness: 0.95 }), treeCount);
const leafI = new THREE.InstancedMesh(new THREE.IcosahedronGeometry(1, 1), new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.85 }), treeCount * 3);
let tn = 0, ln = 0;
const greens = [0x2f7d3a, 0x3d8f45, 0x2a6b36, 0x4f9b3a, 0x6aa84a].map((c) => new THREE.Color(c));
function tree(x, z, s) {
  trunkI.setMatrixAt(tn++, M.compose(P.set(x, 1.7 * s + 0.3, z), Q.identity(), S.set(s, s, s)));
  for (let k = 0; k < 3; k++) {
    const r = (1.9 + rand() * 0.9) * s;
    leafI.setMatrixAt(ln, M.compose(P.set(x + (rand() - 0.5) * 1.6 * s, (4.1 + k * 0.7 + rand() * 0.5) * s, z + (rand() - 0.5) * 1.6 * s), Q.identity(), S.set(r, r * 0.85, r)));
    leafI.setColorAt(ln++, greens[Math.floor(rand() * greens.length)]);
  }
}
for (const [x, z, s] of treePos) tree(x, z, s);
for (const p of parks) {
  const grass = new THREE.Mesh(new THREE.PlaneGeometry(BLOCK - 3, BLOCK - 3), new THREE.MeshStandardMaterial({ color: 0x3b7a3a, roughness: 1 }));
  grass.rotation.x = -Math.PI / 2; grass.position.set(p.cx, 0.32, p.cz); grass.receiveShadow = true; scene.add(grass);
  for (let k = 0; k < 9; k++) tree(p.cx + (rand() - 0.5) * (BLOCK - 10), p.cz + (rand() - 0.5) * (BLOCK - 10), 1 + rand() * 0.7);
}
trunkI.count = tn; leafI.count = ln;
trunkI.castShadow = leafI.castShadow = true;
scene.add(trunkI, leafI);

// road paint: a double yellow line down every road, zebra crossings at every junction
const yellow = [], white = [];
const lineGeo = new THREE.PlaneGeometry(1, 1).rotateX(-Math.PI / 2);
for (let i = 0; i <= CITY; i++) {
  const c = -HALF + i * CELL + ROAD / 2;
  for (let j = 0; j < CITY; j++) {
    const len = BLOCK - 3, mid = -HALF + j * CELL + ROAD + 1.5 + len / 2;
    for (const off of [-0.22, 0.22]) { yellow.push([c + off, mid, 0.14, len]); yellow.push([mid, c + off, len, 0.14]); }
  }
  for (let j = 0; j <= CITY; j++) {
    const k = -HALF + j * CELL + ROAD / 2;
    for (let s = -5; s <= 5; s++) for (const side of [-1, 1]) {
      white.push([c + s * 1.15, k + side * (ROAD / 2 + 1.5), 0.55, 3]);
      white.push([c + side * (ROAD / 2 + 1.5), k + s * 1.15, 3, 0.55]);
    }
  }
}
function paint(list, mat) {
  const im = new THREE.InstancedMesh(lineGeo, mat, list.length);
  list.forEach(([x, z, sx, sz], i) => im.setMatrixAt(i, M.compose(P.set(x, 0.04, z), Q.identity(), S.set(sx, 1, sz))));
  im.receiveShadow = true; scene.add(im);
}
paint(yellow, markYellow); paint(white, markWhite);

// the docks (shipping containers by the water) and hills in the haze
const boxes = new THREE.InstancedMesh(box, new THREE.MeshStandardMaterial({ roughness: 0.55, metalness: 0.5 }), 120);
const cc = [0xc0392b, 0x2e86c1, 0xd4ac0d, 0x229954, 0x7d3c98, 0xe67e22].map((c) => new THREE.Color(c));
for (let i = 0; i < 120; i++) {
  const row = Math.floor(i / 30), col = i % 30;
  boxes.setMatrixAt(i, M.compose(P.set(-HALF + 20 + col * 7.2, 1.6 + (rand() < 0.3 ? 2.6 : 0), HALF + 30 + row * 3.2), Q.identity(), S.set(6.6, 2.6, 2.5)));
  boxes.setColorAt(i, cc[Math.floor(rand() * cc.length)]);
}
boxes.castShadow = boxes.receiveShadow = true;
scene.add(boxes);
const hills = new THREE.Group();
const hillMat = new THREE.MeshStandardMaterial({ color: 0x1d3a3f, roughness: 1, flatShading: true });
for (let i = 0; i < 44; i++) {
  const a = (i / 44) * Math.PI * 2 + rand() * 0.1, r = 2300 + rand() * 500, hgt = 160 + rand() * 360;
  const hill = new THREE.Mesh(new THREE.ConeGeometry(260 + rand() * 260, hgt, 6 + Math.floor(rand() * 3)), hillMat);
  hill.position.set(Math.cos(a) * r, hgt / 2 - 20, Math.sin(a) * r); hill.rotation.y = rand() * 3;
  hills.add(hill);
}
scene.add(hills);

// ---------- cars
const palette = [0xd1202a, 0x1f6feb, 0xf2c200, 0x1c9e5b, 0xe8e8ec, 0x15151a, 0xff7a1a, 0x7a4df0, 0x9aa3ad];
const player = buildCar(stage, { kind: "coupe", color: 0xff2d8a, detail: 1, player: true });
scene.add(player);
const PLAYER_MODEL = null;   // a 3D model the user's Bee made (Mnx Studio, ~/models) drives instead of the car, when the Bee put one in
if (PLAYER_MODEL) {
  import("three/addons/loaders/GLTFLoader.js").then(({ GLTFLoader }) => new GLTFLoader().load(PLAYER_MODEL, (gltf) => {
    const model = gltf.scene;
    const size = new THREE.Box3().setFromObject(model).getSize(new THREE.Vector3());
    model.scale.setScalar(4.4 / Math.max(size.z, size.x, 0.01));
    model.position.y -= new THREE.Box3().setFromObject(model).min.y;
    model.traverse((o) => { if (o.isMesh) o.castShadow = true; });
    Object.values(player.userData.meshes).forEach((m) => { m.visible = false; });
    player.userData.wheels.forEach((w) => { w.parent.visible = false; });
    player.add(model);
  }));
}
const state = { speed: 0, heading: 0, money: 0, night: false, cam: 0, slip: 0, vx: 0, vz: 0, rain: false, quality: stage.level };
const spawn = { x: -HALF + ROAD / 2 + 3.3, z: -HALF + CELL * 3 };
function respawn() { player.position.set(spawn.x, 0, spawn.z); state.speed = 0; state.heading = 0; state.vx = state.vz = 0; }
respawn();

// traffic: cars on the right-hand lane of every road; at a junction some turn
const road = (i) => -HALF + i * CELL + ROAD / 2;                                          // the middle of road number i
const lane = (axis, dir, i) => road(i) + (axis === "x" ? dir : -dir) * 3.3;                // travelling along x sits on +z for dir +1
const yaw = (axis, dir) => (axis === "x" ? (dir > 0 ? Math.PI / 2 : -Math.PI / 2) : dir > 0 ? 0 : Math.PI);
const kinds = ["sedan", "hatch", "coupe", "van", "sedan", "hatch"];
const traffic = [];
for (let k = 0; k < 36; k++) {
  const c = buildCar(stage, { kind: kinds[k % kinds.length], color: palette[k % palette.length], detail: 0 });
  const axis = rand() < 0.5 ? "x" : "z", dir = rand() < 0.5 ? 1 : -1, r = Math.floor(rand() * (CITY + 1));
  const a = c.userData.ai = { axis, dir, s: (rand() - 0.5) * CITY * CELL, lat: lane(axis === "x" ? "x" : "z", dir, r), v: 9 + rand() * 7, vNow: 0, slide: new THREE.Vector3(), turned: -1 };
  a.vNow = a.v;
  c.rotation.y = yaw(axis, dir);
  scene.add(c);
  traffic.push(c);
}
const trafficFor = (l) => (l >= 2 ? 36 : l === 1 ? 18 : 8);
traffic.forEach((c, i) => { c.visible = i < trafficFor(stage.level); });
stage.onQuality((l) => { state.quality = l; traffic.forEach((c, i) => { c.visible = i < trafficFor(l); }); });

// ---------- missions: pick up at the beam, deliver to the next
const beamMat = new THREE.ShaderMaterial({
  uniforms: { time: { value: 0 }, color: { value: new THREE.Color(1.0, 0.78, 0.2) } },
  vertexShader: "varying vec2 vUv; void main(){ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }",
  fragmentShader: "varying vec2 vUv; uniform vec3 color; uniform float time; void main(){ float a = pow(1.0 - vUv.y, 1.6) * 0.8 * (0.75 + 0.25 * sin(vUv.y * 38.0 - time * 3.0)); gl_FragColor = vec4(color * 2.2, a); }",
  transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, side: THREE.DoubleSide,
});
const marker = new THREE.Group();
const beam = new THREE.Mesh(new THREE.CylinderGeometry(3.2, 3.2, 110, 24, 1, true).translate(0, 55, 0), beamMat);
const ring = new THREE.Mesh(new THREE.TorusGeometry(4.6, 0.22, 8, 40).rotateX(Math.PI / 2), new THREE.MeshBasicMaterial({ color: new THREE.Color(2.4, 1.8, 0.4) }));
ring.position.y = 0.5;
marker.add(beam, ring);
scene.add(marker);
const missions = ["Pick up the package", "Deliver it to the docks", "Pick up a passenger", "Drop them downtown", "Grab the parts", "Bring them to the garage"];
let mission = 0;
function placeMarker() {
  const i = Math.floor(rand() * CITY), j = Math.floor(rand() * CITY);
  marker.position.set(road(i), 0, -HALF + j * CELL + CELL / 2);
  $("mission").textContent = missions[mission % missions.length];
}
placeMarker();

function toast(text) {
  const t = $("toast");
  t.textContent = text; t.style.opacity = 1; t.style.transform = "translate(-50%,-50%) scale(1)";
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => { t.style.opacity = 0; t.style.transform = "translate(-50%,-50%) scale(.92)"; }, 1700);
}

// ---------- controls: keyboard and touch
const keys = new Set();
const audio = makeAudio();
addEventListener("keydown", (e) => {
  const k = e.key.toLowerCase();
  keys.add(k); audio.start();
  if (k === "c") state.cam = (state.cam + 1) % 3;
  if (k === "n") stage.setTime(stage.night < 0.5 ? 23.2 : 12.4);
  if (k === "t") { state.rain = !state.rain; stage.setRain(state.rain); toast(state.rain ? "Rain" : "Clear sky"); }
  if (k === "m") toast(audio.toggle() ? "Sound off" : "Sound on");
  if (k === "r") respawn();
  if (k === " ") e.preventDefault();
});
addEventListener("keyup", (e) => keys.delete(e.key.toLowerCase()));
addEventListener("pointerdown", () => audio.start(), { once: true });
for (const b of document.querySelectorAll("#touch button")) {
  const k = b.dataset.k;
  b.addEventListener("pointerdown", (e) => { e.preventDefault(); keys.add(k); audio.start(); });
  for (const ev of ["pointerup", "pointerleave", "pointercancel"]) b.addEventListener(ev, () => keys.delete(k));
}
const down = (...names) => names.some((k) => keys.has(k));

// ---------- physics
function hits(x, z, r = 1.5) {
  if (Math.abs(x) > EDGE - 4 || Math.abs(z) > EDGE - 4) return true;
  for (const s of solids) if (x > s.x0 - r && x < s.x1 + r && z > s.z0 - r && z < s.z1 + r) return true;
  return false;
}
const smoke = new Particles(scene, 500, { size: 1.0 });
const sparks = new Particles(scene, 200, { size: 0.28, additive: true });
const V = new THREE.Vector3(), W = new THREE.Vector3(), UP = new THREE.Vector3(0, 1, 0);
const clock = new THREE.Clock();
const camPos = new THREE.Vector3(), camLook = new THREE.Vector3();
let shake = 0, steerNow = 0, fov = 62, camReady = false;
const burst = (at, n, spread, life = 0.5) => {
  for (let i = 0; i < n; i++) sparks.spawn(at, W.set((Math.random() - 0.5) * spread, 2 + Math.random() * 4, (Math.random() - 0.5) * spread), { life, size: 1, color: [1, 0.75, 0.3], alpha: 1, gravity: 13 });
};

function updatePlayer(dt) {
  const gas = down("w", "arrowup", "gas"), back = down("s", "arrowdown"), hand = down(" ", "brake");
  const left = down("a", "arrowleft", "left"), right = down("d", "arrowright", "right");
  const h = state.heading, fx = Math.sin(h), fz = Math.cos(h), rx = -Math.cos(h), rz = Math.sin(h);
  let vf = state.vx * fx + state.vz * fz, vr = state.vx * rx + state.vz * rz;
  if (gas) vf += (vf < 0 ? 40 : 30 * (1 - Math.min(1, vf / 62))) * dt;
  if (back) vf -= (vf > 1 ? 44 : 18) * dt;
  vf *= 1 - (gas ? 0.12 : 0.45) * dt;
  if (hand) vf *= 1 - 1.6 * dt;
  vr *= Math.exp(-(hand ? 1.6 : 7.5) * (1 - 0.35 * stage.wetAmount) * dt);          // sideways grip; less on a handbrake or in the wet
  steerNow += (((left ? 1 : 0) - (right ? 1 : 0)) - steerNow) * Math.min(1, dt * 9);
  state.heading += steerNow * dt * 2.1 * Math.min(1, Math.abs(vf) / 7) / (1 + Math.abs(vf) / 60) * Math.sign(vf || 1) * (hand ? 1.5 : 1);
  state.vx = fx * vf + rx * vr; state.vz = fz * vf + rz * vr;
  const px = player.position.x, pz = player.position.z;
  let nx = px + state.vx * dt, nz = pz + state.vz * dt, hit = false;
  if (hits(nx, pz)) { state.vx *= -0.25; nx = px; hit = true; }
  if (hits(px, nz)) { state.vz *= -0.25; nz = pz; hit = true; }
  if (hit && Math.abs(vf) > 7) { shake = Math.min(1, Math.abs(vf) / 40); burst(V.set(nx, 0.8, nz), 14, 9); state.vx *= 0.7; state.vz *= 0.7; }
  player.position.x = nx; player.position.z = nz;
  player.rotation.y = state.heading;
  state.speed = vf; state.slip = Math.abs(vr);
  const lean = THREE.MathUtils.clamp(-steerNow * Math.min(1, Math.abs(vf) / 30) * 0.045, -0.06, 0.06);
  const pitch = THREE.MathUtils.clamp(((gas ? -1 : 0) + (back || hand ? 1 : 0)) * 0.012, -0.02, 0.02);
  player.rotation.z += (lean - player.rotation.z) * Math.min(1, dt * 6);
  player.rotation.x += (pitch - player.rotation.x) * Math.min(1, dt * 6);
  animateCar(player, dt, vf, steerNow, (back && vf > 1) || hand);
  if (player.userData.beam) player.userData.beam.intensity = 1800 * stage.night * (1 - 0.3 * stage.rainAmount);
  if ((state.slip > 3.2 || (hand && Math.abs(vf) > 14)) && Math.abs(vf) > 6) {       // tyre smoke while drifting
    for (const s of [-0.85, 0.85]) {
      V.set(s, 0.15, -1.4).applyAxisAngle(UP, state.heading).add(player.position);
      smoke.spawn(V, W.set((Math.random() - 0.5) * 1.2, 0.8 + Math.random(), (Math.random() - 0.5) * 1.2), { life: 1.3, size: 1.3, grow: 4.5, color: [0.82, 0.83, 0.86], alpha: 0.4 });
    }
  }
  if (stage.wetAmount > 0.5 && Math.abs(vf) > 10 && Math.random() < 0.7) {           // spray behind the wheels in the wet
    V.set((Math.random() - 0.5) * 1.6, 0.2, -2).applyAxisAngle(UP, state.heading).add(player.position);
    smoke.spawn(V, W.set(0, 1.2, 0), { life: 0.8, size: 0.9, grow: 3, color: [0.75, 0.8, 0.88], alpha: 0.18 });
  }
}

function updateTraffic(dt) {
  for (let ci = 0; ci < traffic.length; ci++) {
    const c = traffic[ci], a = c.userData.ai;
    if (!c.visible) continue;
    const dx = a.axis === "x" ? a.dir : 0, dz = a.axis === "z" ? a.dir : 0;
    const wx = a.axis === "x" ? a.s : a.lat, wz = a.axis === "x" ? a.lat : a.s;
    let want = a.v;
    const watch = (ox, oz, wide) => {                                                  // slow down behind another car or the player
      const rx = ox - wx, rz = oz - wz, ahead = rx * dx + rz * dz, side = Math.abs(rx * dz - rz * dx);
      if (ahead > 0 && ahead < 14 && side < wide) want = Math.min(want, Math.max(0, (ahead - 5.5) * 1.3));
    };
    for (let oi = 0; oi < traffic.length; oi++) if (oi !== ci && traffic[oi].visible) watch(traffic[oi].position.x, traffic[oi].position.z, 2.4);
    watch(player.position.x, player.position.z, 2.6);
    a.vNow += (want - a.vNow) * Math.min(1, dt * 2.5);
    const before = a.s;
    a.s += a.dir * a.vNow * dt;
    // at a junction some cars turn left or right
    const j = Math.round((a.s + HALF - ROAD / 2) / CELL), jc = road(j);
    if (j >= 0 && j <= CITY && (before - jc) * (a.s - jc) <= 0 && a.turned !== j && rand() < 0.3) {
      a.turned = j;
      const nd = rand() < 0.5 ? 1 : -1, axis = a.axis === "x" ? "z" : "x";
      const was = new THREE.Vector3(wx, 0, wz);
      a.axis = axis; a.dir = nd;
      const oldLat = a.lat;
      a.lat = lane(axis, nd, j);
      a.s = oldLat;
      const now = axis === "x" ? V.set(a.s, 0, a.lat) : V.set(a.lat, 0, a.s);
      a.slide.copy(was).sub(now);
    }
    if (Math.abs(a.s) > HALF + 20) a.s = -Math.sign(a.s) * (HALF + 10);
    a.slide.multiplyScalar(Math.max(0, 1 - dt * 5));
    c.position.set((a.axis === "x" ? a.s : a.lat) + a.slide.x, 0, (a.axis === "x" ? a.lat : a.s) + a.slide.z);
    let d = yaw(a.axis, a.dir) - c.rotation.y;
    d = Math.atan2(Math.sin(d), Math.cos(d));
    c.rotation.y += d * Math.min(1, dt * 7);
    animateCar(c, dt, a.vNow, 0, want < a.vNow - 1);
    if (c.position.distanceTo(player.position) < 3.3) {                                 // a bump
      const away = V.copy(c.position).sub(player.position).setY(0).normalize();
      state.vx -= away.x * 6; state.vz -= away.z * 6; shake = Math.max(shake, 0.4); a.vNow = 0;
      burst(V.copy(player.position).addScaledVector(away, 1.6).setY(0.7), 8, 6, 0.4);
    }
  }
}

// ---------- camera
function updateCamera(dt) {
  const sp = Math.abs(state.speed), h = state.heading, k = Math.min(1, sp / 55), fx = Math.sin(h), fz = Math.cos(h);
  if (state.cam === 2) {                                   // bonnet view
    camera.position.set(player.position.x + fx * 0.5, 1.28, player.position.z + fz * 0.5);
    camera.lookAt(player.position.x + fx * 30, 1.2, player.position.z + fz * 30);
    camera.rotateZ(-steerNow * 0.04);
  } else {
    const back = state.cam === 0 ? 8.5 + k * 3.5 : 22, up = state.cam === 0 ? 3.3 + k * 0.8 : 17;
    camPos.set(player.position.x - fx * back, up, player.position.z - fz * back);
    if (camReady) camera.position.lerp(camPos, 1 - Math.pow(0.0006, dt)); else { camera.position.copy(camPos); camReady = true; }   // the first frame starts at the car
    camera.lookAt(camLook.set(player.position.x + fx * (4 + k * 8), 1.4, player.position.z + fz * (4 + k * 8)));
  }
  if (shake > 0.01) { camera.position.x += (Math.random() - 0.5) * shake * 0.6; camera.position.y += (Math.random() - 0.5) * shake * 0.6; shake *= Math.pow(0.02, dt); }
  fov += ((state.cam === 2 ? 74 : 62) + k * 22 - fov) * Math.min(1, dt * 3);
  if (Math.abs(camera.fov - fov) > 0.05) { camera.fov = fov; camera.updateProjectionMatrix(); }
}

// ---------- HUD and minimap
const city = document.createElement("canvas");
city.width = city.height = 900;
{
  const g = city.getContext("2d"), s = city.width / (EDGE * 2);
  g.fillStyle = "#12263a"; g.fillRect(0, 0, 900, 900);
  g.fillStyle = "#363c47"; g.fillRect(0, 0, 900, 900);
  for (let i = 0; i < CITY; i++) for (let j = 0; j < CITY; j++) {
    const bx = -HALF + i * CELL + ROAD + BLOCK / 2, bz = -HALF + j * CELL + ROAD + BLOCK / 2;
    g.fillStyle = parks.some((p) => p.cx === bx && p.cz === bz) ? "#2f6a37" : "#1b1e27";
    g.fillRect((bx - BLOCK / 2 + EDGE) * s, (bz - BLOCK / 2 + EDGE) * s, BLOCK * s, BLOCK * s);
  }
}
const mapCtx = $("map").getContext("2d");
const MS = 0.5;   // minimap scale: pixels per metre
function drawMap() {
  const size = $("map").width, c = size / 2, th = state.heading - Math.PI;
  mapCtx.clearRect(0, 0, size, size);
  mapCtx.save();
  mapCtx.beginPath(); mapCtx.arc(c, c, c - 2, 0, 7); mapCtx.clip();
  mapCtx.fillStyle = "#0e1822"; mapCtx.fillRect(0, 0, size, size);
  mapCtx.translate(c, c); mapCtx.rotate(th); mapCtx.scale(MS, MS);
  mapCtx.translate(-player.position.x, -player.position.z);
  mapCtx.drawImage(city, -EDGE, -EDGE, EDGE * 2, EDGE * 2);
  mapCtx.restore();
  const pt = (x, z) => { const dx = (x - player.position.x) * MS, dz = (z - player.position.z) * MS; return [c + dx * Math.cos(th) - dz * Math.sin(th), c + dx * Math.sin(th) + dz * Math.cos(th)]; };
  let [mx, my] = pt(marker.position.x, marker.position.z);
  const dd = Math.hypot(mx - c, my - c), lim = c - 10;
  if (dd > lim) { mx = c + ((mx - c) / dd) * lim; my = c + ((my - c) / dd) * lim; }
  mapCtx.fillStyle = "#ffd23f"; mapCtx.shadowColor = "#ffd23f"; mapCtx.shadowBlur = 10;
  mapCtx.beginPath(); mapCtx.arc(mx, my, 6, 0, 7); mapCtx.fill();
  mapCtx.shadowBlur = 0; mapCtx.fillStyle = "rgba(255,255,255,.55)";
  for (const t of traffic) if (t.visible) { const [x, y] = pt(t.position.x, t.position.z); if (Math.hypot(x - c, y - c) < c - 6) mapCtx.fillRect(x - 1.5, y - 1.5, 3, 3); }
  mapCtx.fillStyle = "#ff2d8a"; mapCtx.beginPath(); mapCtx.moveTo(c, c - 9); mapCtx.lineTo(c + 6, c + 7); mapCtx.lineTo(c, c + 3); mapCtx.lineTo(c - 6, c + 7); mapCtx.closePath(); mapCtx.fill();
}

let hudT = 0;
function updateHud(dt) {
  hudT += dt;
  if (hudT < 0.05) return;
  hudT = 0;
  const kmh = Math.round(Math.abs(state.speed) * 3.6);
  $("speed").textContent = kmh;
  $("bar").style.setProperty("--p", Math.min(1, kmh / 200));
  const dx = marker.position.x - player.position.x, dz = marker.position.z - player.position.z;
  $("dist").textContent = `${Math.round(Math.hypot(dx, dz))} m`;
  $("arrow").style.transform = `rotate(${-(Math.atan2(dx, dz) - state.heading)}rad)`;
  const hh = Math.floor(stage.time), mm = Math.floor((stage.time - hh) * 60);
  $("clock").textContent = `${String(hh).padStart(2, "0")}:${String(mm).padStart(2, "0")}${state.rain ? " · rain" : ""}`;
}

// ---------- the loop
let elapsed = 0;
function frame() {
  const dt = Math.min(clock.getDelta(), 0.05);
  elapsed += dt;
  stage.setTime(stage.time + dt * 0.012);                  // a day lasts about half an hour
  state.night = stage.night > 0.5;
  updatePlayer(dt);
  updateTraffic(dt);
  marker.rotation.y += dt; beamMat.uniforms.time.value = elapsed;
  ring.scale.setScalar(1 + Math.sin(elapsed * 4) * 0.06);
  if (Math.hypot(player.position.x - marker.position.x, player.position.z - marker.position.z) < 5.5) {
    const pay = 100 + Math.floor(rand() * 400);
    state.money += pay;
    $("money").textContent = `$${state.money.toLocaleString()}`;
    toast(`${missions[mission % missions.length]}: done  +$${pay}`);
    mission++;
    placeMarker();
  }
  blinkMat.emissiveIntensity = Math.sin(elapsed * 3) > 0.7 ? 4 : 0.1;
  poolMat.opacity = stage.night * 0.85 * (1 - 0.3 * stage.rainAmount);
  water.userData.update(dt);
  hills.position.set(camera.position.x, 0, camera.position.z);
  smoke.update(dt); sparks.update(dt);
  updateCamera(dt);
  stage.follow(player.position);
  stage.update(dt);
  audio.update(Math.min(1, Math.abs(state.speed) / 55), down("w", "arrowup", "gas") ? 1 : 0, Math.min(1, state.slip / 8), stage.rainAmount);
  updateHud(dt);
  drawMap();
  stage.render(dt, Math.min(1, Math.abs(state.speed) / 55));
  requestAnimationFrame(frame);
}

window.__game = { state, player, stage, ready: true };  // lets a Bee check that the game runs
$("loading")?.classList.add("done");
frame();
