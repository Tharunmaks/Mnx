import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";

const canvas = document.getElementById("view");
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x14161d);
const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 2000);
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.autoRotate = true;
controls.autoRotateSpeed = 1.2;
canvas.addEventListener("pointerdown", () => { controls.autoRotate = false; });

scene.add(new THREE.HemisphereLight(0xdfe8ff, 0x30281e, 1.4));
const sun = new THREE.DirectionalLight(0xffffff, 2.2);
sun.position.set(5, 9, 6);
sun.castShadow = true;
sun.shadow.mapSize.set(1024, 1024);
scene.add(sun);
const ground = new THREE.Mesh(new THREE.CircleGeometry(1, 64), new THREE.MeshStandardMaterial({ color: 0x232733, roughness: 1 }));
ground.rotation.x = -Math.PI / 2;
ground.receiveShadow = true;
scene.add(ground);

function resize() {
  const w = innerWidth, h = innerHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
addEventListener("resize", resize);
resize();

new GLTFLoader().load("model.glb", (gltf) => {
  const model = gltf.scene;
  let parts = 0;
  model.traverse((o) => { if (o.isMesh) { o.castShadow = o.receiveShadow = true; parts++; } });
  scene.add(model);
  const box = new THREE.Box3().setFromObject(model);
  const size = box.getSize(new THREE.Vector3());
  const centre = box.getCenter(new THREE.Vector3());
  const r = Math.max(size.x, size.y, size.z) || 1;
  ground.scale.setScalar(r * 1.6);
  ground.position.y = box.min.y - 0.001;
  sun.shadow.camera.left = sun.shadow.camera.bottom = -r * 1.5;
  sun.shadow.camera.right = sun.shadow.camera.top = r * 1.5;
  sun.position.set(centre.x + r * 2, centre.y + r * 3, centre.z + r * 2);
  sun.target.position.copy(centre);
  scene.add(sun.target);
  controls.target.copy(centre);
  camera.position.set(centre.x + r * 1.4, centre.y + r * 0.9, centre.z + r * 1.6);
  camera.near = r / 200; camera.far = r * 50; camera.updateProjectionMatrix();
  document.getElementById("info").textContent = `${parts} parts · ${size.x.toFixed(1)} × ${size.y.toFixed(1)} × ${size.z.toFixed(1)} m · drag to turn, pinch or scroll to zoom`;
  window.__game = { ready: true, parts, size: [size.x, size.y, size.z] };
}, undefined, (err) => { document.getElementById("info").textContent = "Couldn't load the model: " + err; console.error(err); });

renderer.setAnimationLoop(() => { controls.update(); renderer.render(scene, camera); });
