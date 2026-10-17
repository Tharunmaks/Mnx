// Mnx graphics kit: what makes a three.js game look good, in one place. Games import from here.
//   createStage(canvas)  renderer + sky with a sun, moon and stars + clouds + fog + shadows + bloom + tone mapping + rain,
//                        and automatic quality (phones and slow computers get a lighter picture by themselves)
//   buildCar({kind, color, detail})   a car with curved paint, glass, lights, wheels that spin and steer
//   facadeTextures / asphaltTextures / concreteTextures / waterNormal   procedural PBR textures, no downloads
//   Particles, makeWater, makeAudio, rng, lerp
// Nothing here needs files: every texture is drawn when the game starts.
import * as THREE from "three";
import { Sky } from "three/addons/objects/Sky.js";
import { EffectComposer } from "three/addons/postprocessing/EffectComposer.js";
import { RenderPass } from "three/addons/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/addons/postprocessing/UnrealBloomPass.js";
import { OutputPass } from "three/addons/postprocessing/OutputPass.js";
import { ShaderPass } from "three/addons/postprocessing/ShaderPass.js";
import { mergeGeometries } from "three/addons/utils/BufferGeometryUtils.js";

export const lerp = THREE.MathUtils.lerp;
const clamp01 = (x) => Math.min(1, Math.max(0, x));
const smooth = (a, b, x) => { const t = clamp01((x - a) / (b - a)); return t * t * (3 - 2 * t); };
export function rng(seed = 7) { let s = seed % 2147483647 || 1; return () => (s = (s * 16807) % 2147483647) / 2147483647; }

// ====================================================================== the stage
const FINISH = {
  uniforms: { tDiffuse: { value: null }, vignette: { value: 0.28 }, time: { value: 0 }, grain: { value: 0.025 }, aberration: { value: 0 } },
  vertexShader: "varying vec2 vUv; void main(){ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }",
  fragmentShader: `uniform sampler2D tDiffuse; uniform float vignette, time, grain, aberration; varying vec2 vUv;
    float hash(vec2 p){ return fract(sin(dot(p + time, vec2(12.9898, 78.233))) * 43758.5453); }
    void main(){
      vec2 d = vUv - 0.5; vec2 o = d * aberration;
      vec3 c = vec3(texture2D(tDiffuse, vUv + o).r, texture2D(tDiffuse, vUv).g, texture2D(tDiffuse, vUv - o).b);
      float v = smoothstep(0.95, 0.15, length(d) * (1.0 + vignette));
      c *= mix(1.0, v, vignette * 1.5);
      c = mix(vec3(dot(c, vec3(0.299, 0.587, 0.114))), c, 1.16);
      c += (hash(vUv * 1000.0) - 0.5) * grain;
      gl_FragColor = vec4(c, 1.0);
    }`,
};

export function createStage(canvas, opts = {}) {
  const coarse = matchMedia("(pointer: coarse)").matches;
  const qs = new URLSearchParams(location.search);
  const pinned = qs.has("q");                      // ?q=0|1|2 fixes the quality (0 lightest)
  let level = pinned ? Math.max(0, Math.min(2, +qs.get("q") || 0)) : coarse ? 1 : 2;
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: !coarse, powerPreference: "high-performance" });
  renderer.toneMapping = THREE.NeutralToneMapping;     // keeps colours true (ACES turns blue skies grey)
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(62, 1, 0.3, 9000);
  const listeners = [];
  const stage = { renderer, scene, camera, get level() { return level; }, onQuality: (fn) => listeners.push(fn), time: opts.time ?? 17.2,
    night: 0, day: 1, rainAmount: 0, rainTarget: 0, wetAmount: 0, elev: 1 };

  // ---- sky, sun, moon, stars, clouds
  const sky = new Sky();
  sky.scale.setScalar(20000);
  scene.add(sky);
  const skyU = sky.material.uniforms;
  skyU.turbidity.value = 3; skyU.rayleigh.value = 2.2; skyU.mieCoefficient.value = 0.003; skyU.mieDirectionalG.value = 0.82;
  const envScene = new THREE.Scene();
  const skyEnv = new Sky();
  skyEnv.scale.setScalar(20000);
  envScene.add(skyEnv);
  Object.assign(skyEnv.material.uniforms, { turbidity: skyU.turbidity, rayleigh: skyU.rayleigh, mieCoefficient: skyU.mieCoefficient, mieDirectionalG: skyU.mieDirectionalG });
  const pmrem = new THREE.PMREMGenerator(renderer);
  let envRT = null, envElev = 9;

  const hemi = new THREE.HemisphereLight(0xbfdcff, 0x4a4036, 0.8);
  const sun = new THREE.DirectionalLight(0xfff1de, 3);
  sun.castShadow = true;
  Object.assign(sun.shadow.camera, { left: -90, right: 90, top: 90, bottom: -90, near: 10, far: 700 });
  sun.shadow.bias = -0.0004; sun.shadow.normalBias = 0.04;
  const moon = new THREE.DirectionalLight(0x6f8cff, 0);
  scene.add(hemi, sun, sun.target, moon, moon.target);

  const starGeo = new THREE.BufferGeometry();
  const rs = rng(99), sp = [];
  for (let i = 0; i < 1800; i++) { const u = rs() * 2 - 1, a = rs() * 6.2832, r = Math.sqrt(1 - u * u); sp.push(r * Math.cos(a) * 8500, Math.abs(u) * 8500 + 200, r * Math.sin(a) * 8500); }
  starGeo.setAttribute("position", new THREE.Float32BufferAttribute(sp, 3));
  const stars = new THREE.Points(starGeo, new THREE.PointsMaterial({ color: 0xffffff, size: 1.8, sizeAttenuation: false, transparent: true, opacity: 0, fog: false, depthWrite: false }));
  const moonMesh = new THREE.Mesh(new THREE.SphereGeometry(330, 24, 16), new THREE.MeshBasicMaterial({ color: new THREE.Color(2.6, 2.7, 3), fog: false }));
  const clouds = new THREE.Group();
  const cloudTex = softBlobTexture();
  const cloudMat = new THREE.SpriteMaterial({ map: cloudTex, color: 0xffffff, transparent: true, opacity: 0.85, depthWrite: false, fog: false });
  for (let i = 0; i < 9; i++) {
    const s = new THREE.Sprite(cloudMat), a = (i / 9) * 6.2832 + rs() * 0.5, el = 0.2 + rs() * 0.5;
    s.position.set(Math.cos(a) * Math.cos(el) * 7400, Math.sin(el) * 7400, Math.sin(a) * Math.cos(el) * 7400);
    s.scale.set(1700 + rs() * 1700, 380 + rs() * 320, 1);
    clouds.add(s);
  }
  scene.add(stars, moonMesh, clouds);
  scene.fog = new THREE.FogExp2(0xaecbe6, 0.0011);

  // ---- post: bloom, tone mapping, vignette and grain
  let composer = null, bloom = null, finish = null, pr = 1;
  function buildComposer() {
    composer?.dispose?.();
    const rt = new THREE.WebGLRenderTarget(4, 4, { type: THREE.HalfFloatType, samples: level >= 2 && !coarse ? 4 : 0 });
    composer = new EffectComposer(renderer, rt);
    composer.addPass(new RenderPass(scene, camera));
    bloom = new UnrealBloomPass(new THREE.Vector2(512, 512), 0.55, 0.55, 0.92);
    composer.addPass(bloom);
    stage.bloomPass = bloom;
    composer.addPass(new OutputPass());
    finish = new ShaderPass(FINISH);
    composer.addPass(finish);
    composer.setPixelRatio(pr);
    composer.setSize(innerWidth, innerHeight);
  }

  function applyQuality() {
    pr = level >= 2 ? Math.min(devicePixelRatio, 2) : level === 1 ? Math.min(devicePixelRatio, 1.5) : 1;
    renderer.setPixelRatio(pr);
    renderer.shadowMap.enabled = level > 0;
    const size = level >= 2 ? 2048 : 1024;
    if (sun.shadow.map) { sun.shadow.map.dispose(); sun.shadow.map = null; }
    sun.shadow.mapSize.set(size, size);
    sun.castShadow = level > 0;
    if (level > 0) buildComposer(); else { composer?.dispose?.(); composer = null; }
    stage.resize();
    scene.traverse((o) => { if (o.material) { const m = Array.isArray(o.material) ? o.material : [o.material]; m.forEach((x) => { x.needsUpdate = true; }); } });
  }

  stage.resize = () => {
    renderer.setSize(innerWidth, innerHeight, false);
    camera.aspect = innerWidth / innerHeight;
    camera.updateProjectionMatrix();
    composer?.setPixelRatio(pr);
    composer?.setSize(innerWidth, innerHeight);
  };
  addEventListener("resize", stage.resize);

  // ---- time of day (hours, 0 to 24): sun, moon, stars, colours, exposure
  const sunDir = new THREE.Vector3(), moonDir = new THREE.Vector3();
  const warm = new THREE.Color(0xff8a4a), white = new THREE.Color(0xfff3e4);
  const fogDay = new THREE.Color(0xa9c8ea), fogDusk = new THREE.Color(0xe39a6a), fogNight = new THREE.Color(0x070a18), fogRain = new THREE.Color(0x6c7783);
  const tmp = new THREE.Color();
  const glows = [], wets = [];
  stage.glow = (mat, dayI, nightI) => { glows.push({ mat, dayI, nightI }); mat.emissiveIntensity = dayI; return mat; };
  stage.wet = (mat, wetRough = 0.25, wetEnv = 1.6) => { wets.push({ mat, r: mat.roughness, wetRough, env: mat.envMapIntensity ?? 1, wetEnv }); return mat; };
  stage.setRain = (on) => { stage.rainTarget = on ? 1 : 0; };
  stage.setTime = (h) => { stage.time = ((h % 24) + 24) % 24; };

  function applyTime(dt) {
    const phase = ((stage.time - 6) / 24) * Math.PI * 2;
    sunDir.set(Math.cos(phase) * 0.85, Math.sin(phase), 0.42).normalize();
    moonDir.set(-sunDir.x * 0.8, Math.max(0.25, -sunDir.y), -0.4).normalize();
    const elev = sunDir.y, rain = stage.rainAmount;
    const day = smooth(-0.1, 0.22, elev) * (1 - 0.45 * rain);
    const dusk = clamp01(1 - Math.abs(elev) / 0.28) * (1 - rain * 0.6);
    stage.day = day; stage.elev = elev;
    stage.night += ((1 - smooth(-0.14, 0.1, elev)) - stage.night) * Math.min(1, dt * 1.5);
    skyU.sunPosition.value.copy(sunDir); skyEnv.material.uniforms.sunPosition.value.copy(sunDir);
    skyU.rayleigh.value = lerp(0.3, 2.3, day) + dusk * 1.3; skyEnv.material.uniforms.rayleigh.value = skyU.rayleigh.value;
    skyU.turbidity.value = 3 + dusk * 2 + rain * 14; skyEnv.material.uniforms.turbidity.value = skyU.turbidity.value;
    tmp.copy(warm).lerp(white, smooth(0.02, 0.5, elev));
    sun.color.copy(tmp);
    sun.intensity = 3.6 * smooth(-0.02, 0.3, elev) * (1 - 0.7 * rain);
    moon.intensity = 0.55 * (1 - day) * (1 - 0.5 * rain);
    hemi.color.set(0x1a2447).lerp(tmp.set(0xbfdcff), day);
    hemi.groundColor.set(0x0b0c12).lerp(tmp.set(0x4a4036), day);
    hemi.intensity = lerp(0.4, 0.5, day);
    scene.environmentIntensity = lerp(0.38, 0.75, day);
    renderer.toneMappingExposure = lerp(1.5, 0.95, day) * (1 - 0.1 * rain);
    scene.fog.color.copy(fogNight).lerp(fogDay, day).lerp(fogDusk, dusk * 0.8).lerp(fogRain, rain * 0.75);
    scene.fog.density = lerp(0.0016, 0.0008, day) + rain * 0.0018;
    stars.material.opacity = Math.pow(clamp01(1 - day * 1.6), 2) * (1 - rain);
    moonMesh.visible = stage.night > 0.2; moonMesh.position.copy(moonDir).multiplyScalar(7800).add(camera.position);
    cloudMat.color.copy(tmp.set(0xffffff)).lerp(tmp.set(0x0c1226), 1 - day).lerp(tmp.set(0xff9a63), dusk * 0.6).lerp(tmp.set(0x8e96a2), rain * 0.8);
    cloudMat.opacity = lerp(0.3, 0.62, day) + rain * 0.3;
    clouds.position.copy(camera.position);
    clouds.rotation.y += dt * 0.002;
    if (bloom) { bloom.strength = lerp(0.6, 0.28, day); bloom.threshold = lerp(1.0, 1.5, day); }
    if (Math.abs(elev - envElev) > 0.03) {
      envElev = elev;
      envRT?.dispose();
      envRT = pmrem.fromScene(envScene, 0, 1, 40000);
      scene.environment = envRT.texture;
    }
    for (const g of glows) g.mat.emissiveIntensity = lerp(g.dayI, g.nightI, stage.night);
    for (const w of wets) { w.mat.roughness = lerp(w.r, w.r * w.wetRough, stage.wetAmount); w.mat.envMapIntensity = lerp(w.env, w.wetEnv, stage.wetAmount); }
  }

  // ---- rain: streaks that wrap around the camera
  const N = 5000, rp = [], rseed = [], rend = [];
  const rr = rng(5);
  for (let i = 0; i < N; i++) { const a = rr() * 2 - 1, b = rr(), c = rr() * 2 - 1, v = rr(); for (let e = 0; e < 2; e++) { rp.push(0, 0, 0); rseed.push(a, b, c, v); rend.push(e); } }
  const rg = new THREE.BufferGeometry();
  rg.setAttribute("position", new THREE.Float32BufferAttribute(rp, 3));
  rg.setAttribute("aSeed", new THREE.Float32BufferAttribute(rseed, 4));
  rg.setAttribute("aEnd", new THREE.Float32BufferAttribute(rend, 1));
  const rainMat = new THREE.ShaderMaterial({
    uniforms: { time: { value: 0 }, center: { value: new THREE.Vector3() }, opacity: { value: 0 } },
    vertexShader: `attribute vec4 aSeed; attribute float aEnd; uniform float time; uniform vec3 center; varying float vA;
      void main(){ float H = 46.0; float y = mod(aSeed.y * H - time * (30.0 + aSeed.w * 12.0), H) + center.y - 12.0;
        vec2 xz = center.xz + mod(vec2(aSeed.x, aSeed.z) * 45.0 - center.xz + 45.0, 90.0) - 45.0;
        vec3 p = vec3(xz.x, y, xz.y) + aEnd * vec3(-0.25, 1.5, 0.08);
        vA = 1.0 - aEnd * 0.8; gl_Position = projectionMatrix * viewMatrix * vec4(p, 1.0); }`,
    fragmentShader: "uniform float opacity; varying float vA; void main(){ gl_FragColor = vec4(0.75, 0.82, 0.95, vA * opacity); }",
    transparent: true, depthWrite: false,
  });
  const rain = new THREE.LineSegments(rg, rainMat);
  rain.frustumCulled = false; rain.visible = false;
  scene.add(rain);

  // ---- per frame
  let ema = 16, started = performance.now(), hue = 0;
  stage.follow = (target) => {                        // keep the sun's shadow box around the player
    sun.target.position.copy(target); moon.target.position.copy(target);
    sun.position.copy(target).addScaledVector(sunDir, 260);
    moon.position.copy(target).addScaledVector(moonDir, 260);
  };
  stage.update = (dt) => {
    stage.rainAmount += (stage.rainTarget - stage.rainAmount) * Math.min(1, dt * 0.6);
    stage.wetAmount += ((stage.rainTarget ? 1 : 0) - stage.wetAmount) * Math.min(1, dt * (stage.rainTarget ? 0.25 : 0.05));
    applyTime(dt);
    rain.visible = stage.rainAmount > 0.02;
    rainMat.uniforms.time.value += dt; rainMat.uniforms.center.value.copy(camera.position); rainMat.uniforms.opacity.value = 0.38 * stage.rainAmount;
    if (finish) { finish.uniforms.time.value = (hue += dt) % 100; }
  };
  let lastFrame = performance.now(), slowMs = 0;
  stage.render = (dt, speed01 = 0) => {
    if (finish) finish.uniforms.aberration.value = 0.0025 * speed01;
    if (composer) composer.render(dt); else renderer.render(scene, camera);
    const now = performance.now(), real = now - lastFrame;      // real time between frames (dt is capped by the game)
    lastFrame = now;
    ema += (real - ema) * 0.08;                                  // a lighter picture by itself if this device can't keep up
    if (!pinned && level > 0 && now - started > 4000) {
      slowMs = real > 45 ? slowMs + real : Math.max(0, slowMs - real * 2);
      if (slowMs > 2500) { slowMs = 0; ema = 16; level--; applyQuality(); listeners.forEach((fn) => fn(level)); }
    }
  };
  stage.fps = () => 1000 / ema;
  applyQuality();
  applyTime(0);
  return stage;
}

// ====================================================================== textures
function canvasTex(w, h, draw, { srgb = true, repeat = true, aniso = 8 } = {}) {
  const c = document.createElement("canvas"); c.width = w; c.height = h;
  draw(c.getContext("2d"), w, h);
  const t = new THREE.CanvasTexture(c);
  if (srgb) t.colorSpace = THREE.SRGBColorSpace;
  if (repeat) t.wrapS = t.wrapT = THREE.RepeatWrapping;
  t.anisotropy = aniso;
  return t;
}

function softBlobTexture() {
  return canvasTex(256, 128, (g, w, h) => {
    const r = rng(3);
    for (let i = 0; i < 26; i++) {
      const x = w * (0.2 + r() * 0.6), y = h * (0.35 + r() * 0.3), rad = h * (0.22 + r() * 0.32);
      const gr = g.createRadialGradient(x, y, 0, x, y, rad);
      gr.addColorStop(0, "rgba(255,255,255,.35)"); gr.addColorStop(1, "rgba(255,255,255,0)");
      g.fillStyle = gr; g.fillRect(0, 0, w, h);
    }
  }, { repeat: false });
}

export function softDotTexture() {
  return canvasTex(64, 64, (g) => {
    const gr = g.createRadialGradient(32, 32, 0, 32, 32, 32);
    gr.addColorStop(0, "rgba(255,255,255,1)"); gr.addColorStop(0.4, "rgba(255,255,255,.45)"); gr.addColorStop(1, "rgba(255,255,255,0)");
    g.fillStyle = gr; g.fillRect(0, 0, 64, 64);
  }, { repeat: false });
}

/** A tile of building front (8 windows × 8 floors): colour, light-up (night) and roughness/metal maps. Repeat it. */
export function facadeTextures(seed = 1, lit = 0.42) {
  const r = rng(seed * 977), S = 512, cols = 8, rows = 8, cw = S / cols, ch = S / rows;
  const wins = [];
  for (let y = 0; y < rows; y++) for (let x = 0; x < cols; x++) wins.push({ x, y, on: r() < lit, warm: r(), tint: r() });
  const art = (paint) => (g) => {
    paint(g, 0, 0, S, S, null);
    for (const w of wins) paint(g, w.x * cw + cw * 0.16, w.y * ch + ch * 0.2, cw * 0.68, ch * 0.6, w);
  };
  const map = canvasTex(S, S, art((g, x, y, w, h, win) => {
    if (!win) { g.fillStyle = "#c9c9cf"; g.fillRect(0, 0, S, S); for (let i = 0; i < 700; i++) { g.fillStyle = `rgba(${r() < 0.5 ? 255 : 0},${r() < 0.5 ? 255 : 0},${r() < 0.5 ? 255 : 0},.03)`; g.fillRect(r() * S, r() * S, 2 + r() * 6, 2 + r() * 6); }
      g.fillStyle = "rgba(0,0,0,.18)"; for (let i = 0; i < rows; i++) g.fillRect(0, i * ch + ch * 0.92, S, ch * 0.08); return; }
    g.fillStyle = "#1b222c"; g.fillRect(x - 3, y - 3, w + 6, h + 6);
    const gr = g.createLinearGradient(x, y, x, y + h); gr.addColorStop(0, "#5f7fa3"); gr.addColorStop(0.5, "#2c3f57"); gr.addColorStop(1, "#16202d");
    g.fillStyle = gr; g.fillRect(x, y, w, h);
  }));
  const emissive = canvasTex(S, S, art((g, x, y, w, h, win) => {
    if (!win) { g.fillStyle = "#000"; g.fillRect(0, 0, S, S); return; }
    if (!win.on) return;
    g.fillStyle = win.warm < 0.65 ? `hsl(${38 + win.tint * 14},95%,${58 + win.tint * 14}%)` : win.warm < 0.88 ? "#cfe6ff" : "#ff9acb";
    g.fillRect(x, y, w, h);
  }));
  const orm = canvasTex(S, S, art((g, x, y, w, h, win) => {   // green = roughness, blue = metal
    if (!win) { g.fillStyle = "rgb(0,205,0)"; g.fillRect(0, 0, S, S); return; }
    g.fillStyle = "rgb(0,35,230)"; g.fillRect(x - 3, y - 3, w + 6, h + 6);
  }), { srgb: false });
  return { map, emissive, orm, tile: { x: cols * 3.3, y: rows * 3.5 } };   // one tile ≈ this many metres wide and tall
}

export function asphaltTextures() {
  const r = rng(21);
  const map = canvasTex(512, 512, (g, w, h) => {
    g.fillStyle = "#34363c"; g.fillRect(0, 0, w, h);
    for (let i = 0; i < 9000; i++) { const v = 40 + r() * 70; g.fillStyle = `rgba(${v},${v},${v + 4},${0.15 + r() * 0.25})`; g.fillRect(r() * w, r() * h, 1 + r() * 2.5, 1 + r() * 2.5); }
    g.strokeStyle = "rgba(10,10,12,.5)"; g.lineWidth = 1;
    for (let i = 0; i < 7; i++) { g.beginPath(); let x = r() * w, y = r() * h; g.moveTo(x, y); for (let k = 0; k < 9; k++) { x += (r() - 0.5) * 60; y += (r() - 0.3) * 50; g.lineTo(x, y); } g.stroke(); }
    for (let i = 0; i < 5; i++) { g.fillStyle = "rgba(0,0,0,.10)"; g.beginPath(); g.ellipse(r() * w, r() * h, 20 + r() * 50, 10 + r() * 24, r() * 3, 0, 6.3); g.fill(); }
  });
  const rough = canvasTex(256, 256, (g, w, h) => {
    g.fillStyle = "rgb(0,170,0)"; g.fillRect(0, 0, w, h);
    for (let i = 0; i < 2500; i++) { g.fillStyle = `rgba(0,${100 + r() * 130 | 0},0,.5)`; g.fillRect(r() * w, r() * h, 2 + r() * 12, 2 + r() * 12); }
  }, { srgb: false });
  return { map, rough };
}

export function concreteTextures() {
  const r = rng(33);
  const map = canvasTex(256, 256, (g, w, h) => {
    g.fillStyle = "#8d9096"; g.fillRect(0, 0, w, h);
    for (let i = 0; i < 4000; i++) { const v = 100 + r() * 70; g.fillStyle = `rgba(${v},${v},${v + 3},.18)`; g.fillRect(r() * w, r() * h, 1 + r() * 3, 1 + r() * 3); }
    g.strokeStyle = "rgba(40,42,48,.55)"; g.lineWidth = 2; g.strokeRect(1, 1, w - 2, h - 2);
  });
  return { map };
}

/** A tileable normal map for water, from a few crossing waves. */
export function waterNormal(size = 256) {
  const hgt = new Float32Array(size * size), waves = [[3, 1, 1], [-2, 5, 0.7], [6, -3, 0.45], [1, 9, 0.3], [-8, -4, 0.22]];
  for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) {
    let v = 0;
    for (const [kx, ky, a] of waves) v += a * Math.sin(((x * kx + y * ky) / size) * 6.2832 + kx * 1.7);
    hgt[y * size + x] = v;
  }
  const c = document.createElement("canvas"); c.width = c.height = size;
  const g = c.getContext("2d"), img = g.createImageData(size, size);
  for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) {
    const dx = hgt[y * size + ((x + 1) % size)] - hgt[y * size + ((x - 1 + size) % size)];
    const dy = hgt[((y + 1) % size) * size + x] - hgt[((y - 1 + size) % size) * size + x];
    const n = new THREE.Vector3(-dx * 0.55, -dy * 0.55, 1).normalize(), i = (y * size + x) * 4;
    img.data[i] = (n.x * 0.5 + 0.5) * 255; img.data[i + 1] = (n.y * 0.5 + 0.5) * 255; img.data[i + 2] = (n.z * 0.5 + 0.5) * 255; img.data[i + 3] = 255;
  }
  g.putImageData(img, 0, 0);
  const t = new THREE.CanvasTexture(c);
  t.wrapS = t.wrapT = THREE.RepeatWrapping; t.anisotropy = 8;
  return t;
}

/** Sea or lake: reflects the sky, with moving waves. Call water.userData.update(dt) every frame. */
export function makeWater(size = 12000, y = -0.6) {
  const nm = waterNormal();
  nm.repeat.set(size / 60, size / 60);
  const mat = new THREE.MeshStandardMaterial({ color: 0x0b4766, roughness: 0.06, metalness: 0.85, normalMap: nm, normalScale: new THREE.Vector2(0.55, 0.55) });
  const m = new THREE.Mesh(new THREE.PlaneGeometry(size, size), mat);
  m.rotation.x = -Math.PI / 2; m.position.y = y;
  m.userData.update = (dt) => { nm.offset.x += dt * 0.006; nm.offset.y += dt * 0.0035; };
  return m;
}

/** A glow on the ground under a lamp (night only: add it to a stage.glow-driven material). */
export function lightPoolTexture() {
  return canvasTex(128, 128, (g) => {
    const gr = g.createRadialGradient(64, 64, 0, 64, 64, 64);
    gr.addColorStop(0, "rgba(255,214,150,.75)"); gr.addColorStop(0.5, "rgba(255,190,110,.22)"); gr.addColorStop(1, "rgba(255,170,90,0)");
    g.fillStyle = gr; g.fillRect(0, 0, 128, 128);
  }, { repeat: false });
}

// ====================================================================== particles (smoke, sparks, dust)
export class Particles {
  constructor(scene, max = 400, { size = 1, additive = false } = {}) {
    this.max = max; this.live = [];
    this.geo = new THREE.BufferGeometry();
    this.pos = new Float32Array(max * 3); this.col = new Float32Array(max * 4); this.siz = new Float32Array(max);
    this.geo.setAttribute("position", new THREE.BufferAttribute(this.pos, 3));
    this.geo.setAttribute("aColor", new THREE.BufferAttribute(this.col, 4));
    this.geo.setAttribute("aSize", new THREE.BufferAttribute(this.siz, 1));
    this.mat = new THREE.ShaderMaterial({
      uniforms: { map: { value: softDotTexture() }, scale: { value: innerHeight * 0.5 * size } },
      vertexShader: `attribute vec4 aColor; attribute float aSize; uniform float scale; varying vec4 vC;
        void main(){ vC = aColor; vec4 mv = modelViewMatrix * vec4(position, 1.0); gl_PointSize = aSize * scale / -mv.z; gl_Position = projectionMatrix * mv; }`,
      fragmentShader: "uniform sampler2D map; varying vec4 vC; void main(){ vec4 t = texture2D(map, gl_PointCoord); gl_FragColor = vec4(vC.rgb, vC.a * t.a); }",
      transparent: true, depthWrite: false, blending: additive ? THREE.AdditiveBlending : THREE.NormalBlending,
    });
    this.points = new THREE.Points(this.geo, this.mat);
    this.points.frustumCulled = false;
    scene.add(this.points);
  }
  spawn(p, v, { life = 1, size = 1, grow = 1, color = [1, 1, 1], alpha = 0.5, gravity = 0 } = {}) {
    if (this.live.length >= this.max) this.live.shift();
    this.live.push({ p: p.clone(), v: v.clone(), t: 0, life, size, grow, color, alpha, gravity });
  }
  update(dt) {
    const L = this.live;
    for (let i = L.length - 1; i >= 0; i--) { const q = L[i]; q.t += dt; if (q.t >= q.life) L.splice(i, 1); else { q.v.y -= q.gravity * dt; q.p.addScaledVector(q.v, dt); } }
    for (let i = 0; i < this.max; i++) {
      const q = L[i];
      if (!q) { this.siz[i] = 0; this.col[i * 4 + 3] = 0; continue; }
      const k = q.t / q.life;
      this.pos.set([q.p.x, q.p.y, q.p.z], i * 3);
      this.siz[i] = q.size * (1 + (q.grow - 1) * k);
      this.col.set([q.color[0], q.color[1], q.color[2], q.alpha * (1 - k) * Math.min(1, k * 8)], i * 4);
    }
    this.geo.attributes.position.needsUpdate = this.geo.attributes.aColor.needsUpdate = this.geo.attributes.aSize.needsUpdate = true;
  }
}

// ====================================================================== cars
const SHARED = {};
function shared(stage) {
  if (SHARED.stage === stage) return SHARED;
  Object.assign(SHARED, { stage });
  SHARED.glass = new THREE.MeshPhysicalMaterial({ color: 0x0a1118, metalness: 0.85, roughness: 0.04, clearcoat: 1, clearcoatRoughness: 0.02, envMapIntensity: 1.5 });
  SHARED.black = new THREE.MeshStandardMaterial({ color: 0x0d0d10, roughness: 0.7, metalness: 0.2 });
  SHARED.tire = new THREE.MeshStandardMaterial({ color: 0x101012, roughness: 0.92, metalness: 0 });
  SHARED.chrome = new THREE.MeshStandardMaterial({ color: 0xd9dde3, roughness: 0.18, metalness: 1 });
  SHARED.rim = new THREE.MeshStandardMaterial({ color: 0xb9bec7, roughness: 0.22, metalness: 1 });
  SHARED.caliper = new THREE.MeshStandardMaterial({ color: 0xc8161d, roughness: 0.4, metalness: 0.4 });
  SHARED.lamp = new THREE.MeshStandardMaterial({ color: 0xffffff, emissive: 0xfff2d2, roughness: 0.2 });
  SHARED.tail = new THREE.MeshStandardMaterial({ color: 0x300004, emissive: 0xff1822, roughness: 0.3 });
  SHARED.plate = new THREE.MeshStandardMaterial({ color: 0xf2f2ee, roughness: 0.5 });
  stage.glow(SHARED.lamp, 0.8, 3.2);
  stage.glow(SHARED.tail, 0.5, 1.8);
  SHARED.paint = new Map();
  return SHARED;
}
function paintMat(S, color) {
  const key = new THREE.Color(color).getHexString();
  if (!S.paint.has(key)) S.paint.set(key, new THREE.MeshPhysicalMaterial({ color, metalness: 0.55, roughness: 0.3, clearcoat: 1, clearcoatRoughness: 0.04, envMapIntensity: 1.2 }));
  return S.paint.get(key);
}

const KINDS = {   // length, width, belt height, hood drop, trunk drop, cabin (front x, roof front x, roof rear x, rear x, roof y)
  coupe: { L: 4.4, W: 1.86, belt: 0.98, hood: 0.28, trunk: 0.08, cab: [0.9, 0.34, -0.62, -1.18, 1.43], spoiler: true },
  sedan: { L: 4.7, W: 1.82, belt: 1.0, hood: 0.2, trunk: 0.1, cab: [1.0, 0.45, -0.85, -1.4, 1.5] },
  hatch: { L: 3.9, W: 1.74, belt: 1.02, hood: 0.18, trunk: 0.02, cab: [0.95, 0.5, -1.0, -1.5, 1.56] },
  van: { L: 4.9, W: 1.95, belt: 1.18, hood: 0.34, trunk: 0.0, cab: [1.45, 1.05, -2.0, -2.3, 2.1] },
};

function bodyShape(k) {
  const f = k.L / 2, b = -f, s = new THREE.Shape(), wy = 0.36, ar = 0.46;
  const fw = f - 0.72, rw = b + 0.78;                 // wheel centres
  s.moveTo(b + 0.06, 0.3);
  s.lineTo(b, 0.58);
  s.bezierCurveTo(b - 0.02, k.belt - 0.12, b + 0.2, k.belt - k.trunk, b + 0.7, k.belt - k.trunk * 0.5);
  s.lineTo(b + 1.5, k.belt);
  s.lineTo(f - 1.1, k.belt + 0.02);
  s.bezierCurveTo(f - 0.5, k.belt, f - 0.2, k.belt - k.hood, f, 0.72 - k.hood * 0.2);
  s.lineTo(f + 0.02, 0.4);
  s.lineTo(f - 0.1, 0.22);
  const a0 = Math.asin((0.2 - wy) / ar);
  s.lineTo(fw + ar * Math.cos(a0), 0.2);
  s.absarc(fw, wy, ar, a0, Math.PI - a0, false);
  s.lineTo(rw + ar * Math.cos(a0), 0.2);
  s.absarc(rw, wy, ar, a0, Math.PI - a0, false);
  s.lineTo(b + 0.16, 0.2);
  return { shape: s, fw, rw, wy };
}

function cabinShape(k) {
  const [fx, rfx, rrx, bx, ry] = k.cab, s = new THREE.Shape();
  s.moveTo(bx, k.belt - 0.04);
  s.bezierCurveTo(bx + 0.1, k.belt + 0.2, rrx - 0.2, ry - 0.02, rrx, ry);
  s.lineTo(rfx, ry);
  s.bezierCurveTo(rfx + 0.28, ry - 0.02, fx - 0.3, k.belt + 0.2, fx, k.belt - 0.04);
  return s;
}

function extruded(shape, depth, bevel, segs = 4) {
  const g = new THREE.ExtrudeGeometry(shape, { depth, bevelEnabled: true, bevelThickness: bevel, bevelSize: bevel, bevelSegments: segs, curveSegments: 18 });
  g.rotateY(-Math.PI / 2);                            // shape x (length) → world z, extrusion → world x
  g.computeBoundingBox();
  g.translate(-(g.boundingBox.min.x + g.boundingBox.max.x) / 2, 0, 0);
  return g;
}

/** A car facing +z, standing on y = 0. detail 0 = cheap (traffic), 1 = full. userData: wheels, steer, brake(on), lights. */
export function buildCar(stage, { kind = "coupe", color = 0xd1202a, detail = 1, player = false } = {}) {
  const S = shared(stage), k = KINDS[kind] || KINDS.coupe, group = new THREE.Group();
  const paint = paintMat(S, color);
  const parts = { paint: [], glass: [], black: [], chrome: [], lamp: [], tail: [], plate: [] };
  const add = (key, geo, x = 0, y = 0, z = 0, rx = 0, ry = 0, rz = 0) => {
    const g = geo.index ? geo.toNonIndexed() : geo.clone();
    g.applyMatrix4(new THREE.Matrix4().compose(new THREE.Vector3(x, y, z), new THREE.Quaternion().setFromEuler(new THREE.Euler(rx, ry, rz)), new THREE.Vector3(1, 1, 1)));
    parts[key].push(g);
  };
  const { shape, fw, rw, wy } = bodyShape(k);
  const bw = k.W - 0.2;
  const body = extruded(shape, bw, 0.1, 5);
  add("paint", body);
  const cab = extruded(cabinShape(k), bw - 0.28, 0.1, 3);
  add("glass", cab);
  const [, rfx, rrx, , ry] = k.cab;
  add("paint", new THREE.BoxGeometry(bw - 0.12, 0.05, rfx - rrx + 0.1), 0, ry + 0.07, (rfx + rrx) / 2);   // roof panel
  add("black", new THREE.BoxGeometry(0.95, 0.2, 0.08), 0, 0.56, k.L / 2 + 0.05);                          // grille
  add("lamp", new THREE.BoxGeometry(0.42, 0.1, 0.1), -0.62, 0.74, k.L / 2 - 0.05, 0, 0.1, 0);
  add("lamp", new THREE.BoxGeometry(0.42, 0.1, 0.1), 0.62, 0.74, k.L / 2 - 0.05, 0, -0.1, 0);
  add("tail", new THREE.BoxGeometry(bw - 0.35, 0.1, 0.08), 0, k.belt - 0.2, -k.L / 2 + 0.01);
  add("plate", new THREE.BoxGeometry(0.5, 0.13, 0.03), 0, 0.5, -k.L / 2 - 0.15);
  add("black", new THREE.BoxGeometry(bw + 0.1, 0.12, 0.16), 0, 0.26, k.L / 2 - 0.08);                    // front splitter
  add("black", new THREE.BoxGeometry(bw - 0.2, 0.14, 0.14), 0, 0.28, -k.L / 2 + 0.1);                    // rear diffuser
  if (detail > 0) {
    for (const sx of [-1, 1]) {
      add("paint", new THREE.BoxGeometry(0.1, 0.1, 0.2), sx * (bw / 2 + 0.2), k.belt + 0.02, rfx + 0.2);   // mirrors
      add("black", new THREE.BoxGeometry(0.02, 0.4, 0.02), sx * (bw / 2 + 0.2), 0.7, 0.6);                  // door lines
      add("black", new THREE.BoxGeometry(0.02, 0.4, 0.02), sx * (bw / 2 + 0.2), 0.7, -0.5);
      add("chrome", new THREE.CylinderGeometry(0.05, 0.05, 0.22, 12), sx * 0.45, 0.33, -k.L / 2 - 0.05, Math.PI / 2, 0, 0);   // exhausts
      add("lamp", new THREE.BoxGeometry(0.34, 0.025, 0.04), sx * 0.58, 0.82, k.L / 2 - 0.02);               // daytime strips
    }
    add("plate", new THREE.BoxGeometry(0.5, 0.13, 0.03), 0, 0.46, k.L / 2 + 0.07);
    if (k.spoiler) {
      add("black", new THREE.BoxGeometry(bw - 0.1, 0.05, 0.3), 0, k.belt + 0.15, -k.L / 2 + 0.3);
      for (const sx of [-1, 1]) add("black", new THREE.BoxGeometry(0.05, 0.16, 0.12), sx * 0.6, k.belt + 0.04, -k.L / 2 + 0.32);
    }
  }
  const meshes = {};
  const mats = { paint, glass: S.glass, black: S.black, chrome: S.chrome, lamp: S.lamp, tail: player ? S.tail.clone() : S.tail, plate: S.plate };
  if (player) stage.glow(mats.tail, 0.5, 1.8);

  // wheels
  const wheels = [], steer = [];
  const tireG = new THREE.CylinderGeometry(0.36, 0.36, 0.27, detail ? 28 : 12).rotateZ(Math.PI / 2);
  const rimParts = [];
  const rimG = new THREE.CylinderGeometry(0.245, 0.245, 0.285, detail ? 22 : 8).rotateZ(Math.PI / 2);
  const spokeG = new THREE.BoxGeometry(0.29, 0.05, 0.4);
  if (detail) {
    for (const [wx, wz, front] of [[-1, fw, true], [1, fw, true], [-1, rw, false], [1, rw, false]]) {
      const pivot = new THREE.Group(), spin = new THREE.Group();
      pivot.position.set(wx * (k.W / 2 - 0.1), wy, wz);
      spin.add(new THREE.Mesh(tireG, S.tire), new THREE.Mesh(rimG, S.rim));
      for (let i = 0; i < 5; i++) { const sp = new THREE.Mesh(spokeG, S.black); sp.rotation.x = (i / 5) * Math.PI * 2; spin.add(sp); }
      const cal = new THREE.Mesh(new THREE.BoxGeometry(0.06, 0.2, 0.12), S.caliper); cal.position.set(wx * -0.09, 0.2, 0); spin.add(cal);
      spin.traverse((o) => { o.castShadow = true; });
      pivot.add(spin); group.add(pivot);
      wheels.push(spin); steer.push(...(front ? [pivot] : []));
    }
  } else {                                           // traffic: the tyres are part of the one dark mesh (fewer draw calls)
    for (const [wx, wz] of [[-1, fw], [1, fw], [-1, rw], [1, rw]]) {
      const g = tireG.clone(); g.translate(wx * (k.W / 2 - 0.1), wy, wz);
      parts.black.push(g.toNonIndexed());
    }
  }
  { const l = new THREE.CylinderGeometry(0.44, 0.44, k.W - 0.5, 14).rotateZ(Math.PI / 2);   // inside the arches, so you can't see through
    for (const z of [fw, rw]) parts.black.push(l.clone().translate(0, wy, z).toNonIndexed()); }
  for (const [key, list] of Object.entries(parts)) {
    if (!list.length || (key === "chrome" && !detail && !list.length)) continue;
    const m = new THREE.Mesh(mergeGeometries(list), mats[key]);
    m.castShadow = key !== "lamp" && key !== "tail"; m.receiveShadow = false;
    group.add(m); meshes[key] = m;
  }

  // headlight beams (the player's car; they follow the night)
  let beam = null;
  if (player) {
    beam = new THREE.SpotLight(0xfff0d6, 0, 140, 0.55, 0.6, 1.3);
    beam.position.set(0, 0.8, k.L / 2 - 0.2); beam.target.position.set(0, 0.2, k.L / 2 + 30);
    group.add(beam, beam.target);
  }
  group.userData = { wheels, steer, kind, length: k.L, width: k.W, tail: mats.tail, meshes, beam };
  return group;
}

/** Spin, steer and settle the car's body: speed in m/s (forward +), steer -1..1, brake bool. */
export function animateCar(car, dt, speed, steer = 0, brake = false) {
  const u = car.userData;
  for (const w of u.wheels) w.rotation.x += (speed / 0.36) * dt;
  for (const p of u.steer) p.rotation.y += (steer * 0.5 - p.rotation.y) * Math.min(1, dt * 10);
  if (u.tail && u.tail.userData.brake !== brake) { u.tail.userData.brake = brake; u.tail.color.setHex(brake ? 0xaa000a : 0x300004); }
}

// ====================================================================== sound (no files: a synthesised engine)
export function makeAudio() {
  let ctx = null, osc, osc2, gain, filt, noise, nGain, muted = false;
  const start = () => {
    if (ctx) return;
    try {
      ctx = new (window.AudioContext || window.webkitAudioContext)();
      gain = ctx.createGain(); gain.gain.value = 0;
      filt = ctx.createBiquadFilter(); filt.type = "lowpass"; filt.frequency.value = 600;
      osc = ctx.createOscillator(); osc.type = "sawtooth"; osc2 = ctx.createOscillator(); osc2.type = "square";
      osc.connect(filt); osc2.connect(filt); filt.connect(gain); gain.connect(ctx.destination);
      osc.start(); osc2.start();
      const buf = ctx.createBuffer(1, ctx.sampleRate * 2, ctx.sampleRate), d = buf.getChannelData(0);
      for (let i = 0; i < d.length; i++) d[i] = Math.random() * 2 - 1;
      noise = ctx.createBufferSource(); noise.buffer = buf; noise.loop = true;
      const nf = ctx.createBiquadFilter(); nf.type = "bandpass"; nf.frequency.value = 1800;
      nGain = ctx.createGain(); nGain.gain.value = 0;
      noise.connect(nf); nf.connect(nGain); nGain.connect(ctx.destination); noise.start();
    } catch { ctx = null; }
  };
  return {
    start,
    toggle() { muted = !muted; return muted; },
    update(speed01, throttle, slip = 0, rain = 0) {
      if (!ctx) return;
      const t = ctx.currentTime, rpm = 38 + speed01 * 150 + throttle * 28;
      osc.frequency.setTargetAtTime(rpm, t, 0.05); osc2.frequency.setTargetAtTime(rpm * 0.5, t, 0.05);
      filt.frequency.setTargetAtTime(380 + speed01 * 1500 + throttle * 600, t, 0.08);
      gain.gain.setTargetAtTime(muted ? 0 : 0.035 + throttle * 0.03 + speed01 * 0.02, t, 0.08);
      nGain.gain.setTargetAtTime(muted ? 0 : Math.min(0.12, slip * 0.12 + rain * 0.025), t, 0.1);
    },
  };
}
