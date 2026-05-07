from __future__ import annotations

import streamlit as st


APP_HTML = r"""
<div id="app">
  <aside class="panel">
    <h1>Three-Body Simulation</h1>

    <label>Preset</label>
    <div class="segmented">
      <button id="preset-random" class="active" type="button">Random</button>
      <button id="preset-solar" type="button">Solar</button>
    </div>

    <label for="bodyCount">Bodies <span id="bodyCountValue">3</span></label>
    <input id="bodyCount" type="range" min="1" max="8" step="1" value="3">

    <label for="gravity">Gravity <span id="gravityValue">1.0</span></label>
    <input id="gravity" type="range" min="0.1" max="5" step="0.1" value="1">

    <label>Sandbox</label>
    <div class="triple">
      <input id="limitX" type="number" min="2" max="30" step="0.5" value="12">
      <input id="limitY" type="number" min="2" max="30" step="0.5" value="12">
      <input id="limitZ" type="number" min="2" max="30" step="0.5" value="12">
    </div>

    <div class="switches">
      <label class="check"><input id="showGrid" type="checkbox"> Grid</label>
      <label class="check"><input id="showAxisNumbers" type="checkbox"> Axis numbers</label>
    </div>

    <label for="stepsPerFrame">Steps/frame <span id="stepsPerFrameValue">6</span></label>
    <input id="stepsPerFrame" type="range" min="1" max="80" step="1" value="6">

    <label for="trailLength">Trail length <span id="trailLengthValue">650</span></label>
    <input id="trailLength" type="range" min="20" max="2500" step="10" value="650">

    <div class="actions">
      <button id="toggle" type="button">Pause</button>
      <button id="restart" type="button">Restart</button>
    </div>
    <button id="randomize" class="wide" type="button">Refresh random state</button>
  </aside>

  <main class="stage">
    <div class="metrics">
      <div class="metric status-card"><span>Status</span><strong id="status">Running</strong></div>
      <div class="metric"><span>Step</span><strong id="step">0</strong></div>
      <div class="metric"><span>Time</span><strong id="simTime">0.000</strong></div>
      <div class="metric"><span>Bodies</span><strong id="bodyMetric">3</strong></div>
    </div>
    <div id="viewport">
      <div id="indicator"></div>
    </div>
  </main>
</div>

<script type="importmap">
{
  "imports": {
    "three": "https://unpkg.com/three@0.160.0/build/three.module.js",
    "three/addons/": "https://unpkg.com/three@0.160.0/examples/jsm/"
  }
}
</script>

<script type="module">
import * as THREE from "three";
import { TrackballControls } from "three/addons/controls/TrackballControls.js";

const COLORS = [0xe63946, 0x1d3557, 0x2a9d8f, 0xf4a261, 0x8d99ae, 0x264653, 0xe9c46a, 0x118ab2];
const NAMES = ["A", "B", "C", "D", "E", "F", "G", "H"];
const SOLAR_NAMES = ["Sun", "Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus"];
const SOLAR_MASSES = [1.0, 1.66e-7, 2.45e-6, 3.00e-6, 3.23e-7, 9.54e-4, 2.86e-4, 4.37e-5];
const SOLAR_RADII = [0.0, 0.39, 0.72, 1.00, 1.52, 5.20, 9.58, 19.20];
const SOLAR_PHASES = [0.0, 0.3, 1.2, 2.0, 2.7, 3.5, 4.2, 5.0];

const state = {
  preset: "Random",
  baseSeed: Math.floor((Date.now() ^ (Math.random() * 0xffffffff)) >>> 0),
  seedBump: 0,
  running: true,
  step: 0,
  time: 0,
  bodies: [],
  trails: [],
  escaped: false,
  stopReason: "",
};

const viewport = document.getElementById("viewport");
const scene = new THREE.Scene();
scene.background = new THREE.Color(0xf8f5ef);

const camera = new THREE.PerspectiveCamera(48, 1, 0.01, 10000);
camera.position.set(13, 12, 9);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
viewport.appendChild(renderer.domElement);

const controls = new TrackballControls(camera, renderer.domElement);
controls.rotateSpeed = 3.6;
controls.zoomSpeed = 1.15;
controls.panSpeed = 0.75;
controls.staticMoving = false;
controls.dynamicDampingFactor = 0.12;

function usesTouchControls() {
  return window.matchMedia("(pointer: coarse)").matches || navigator.maxTouchPoints > 0;
}

function configureControls() {
  controls.noPan = usesTouchControls();
  if (controls.noPan) controls.target.set(0, 0, 0);
}

const bodyGroup = new THREE.Group();
const trailGroup = new THREE.Group();
const boundsGroup = new THREE.Group();
const helperGroup = new THREE.Group();
scene.add(helperGroup, trailGroup, bodyGroup, boundsGroup);

scene.add(new THREE.HemisphereLight(0xffffff, 0x2b3431, 2.2));
const keyLight = new THREE.DirectionalLight(0xffffff, 2.4);
keyLight.position.set(8, 12, 10);
scene.add(keyLight);

const ui = Object.fromEntries(
  ["bodyCount", "gravity", "limitX", "limitY", "limitZ", "showGrid", "showAxisNumbers", "stepsPerFrame", "trailLength"].map((id) => [id, document.getElementById(id)])
);

function rand(seed) {
  let value = seed >>> 0;
  return () => {
    value = (1664525 * value + 1013904223) >>> 0;
    return value / 4294967296;
  };
}

function normal(rng) {
  const u = Math.max(1e-12, rng());
  const v = Math.max(1e-12, rng());
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function v(x = 0, y = 0, z = 0) {
  return new THREE.Vector3(x, y, z);
}

function randomDirection(rng) {
  const vec = v(normal(rng), normal(rng), normal(rng));
  return vec.lengthSq() > 1e-12 ? vec.normalize() : v(1, 0, 0);
}

function readConfig() {
  return {
    bodyCount: Number(ui.bodyCount.value),
    dt: 0.003,
    gravity: Number(ui.gravity.value),
    softening: 0.00001,
    limitX: Number(ui.limitX.value),
    limitY: Number(ui.limitY.value),
    limitZ: Number(ui.limitZ.value),
    showGrid: ui.showGrid.checked,
    showAxisNumbers: ui.showAxisNumbers.checked,
    stepsPerFrame: Number(ui.stepsPerFrame.value),
    trailLength: Number(ui.trailLength.value),
    seed: state.baseSeed + state.seedBump,
  };
}

function updateLabels() {
  document.getElementById("bodyCountValue").textContent = ui.bodyCount.value;
  document.getElementById("gravityValue").textContent = Number(ui.gravity.value).toFixed(1);
  document.getElementById("stepsPerFrameValue").textContent = ui.stepsPerFrame.value;
  document.getElementById("trailLengthValue").textContent = ui.trailLength.value;
  document.getElementById("bodyMetric").textContent = state.bodies.length;
  document.getElementById("step").textContent = state.step.toLocaleString();
  document.getElementById("simTime").textContent = state.time.toFixed(3);
  document.getElementById("status").textContent = state.stopReason || (state.running ? "Running" : "Paused");
  updateIndicator();
}

function updateIndicator() {
  const indicator = document.getElementById("indicator");
  indicator.innerHTML = "";
  for (let i = 0; i < state.bodies.length; i++) {
    const body = state.bodies[i];
    const speed = body.velocity.length();
    const azimuth = THREE.MathUtils.radToDeg(Math.atan2(body.velocity.y, body.velocity.x));
    const elevation = THREE.MathUtils.radToDeg(Math.atan2(body.velocity.z, Math.hypot(body.velocity.x, body.velocity.y)));
    const row = document.createElement("div");
    row.className = "indicator-row";
    row.innerHTML = `
      <span class="indicator-dot" style="background:#${COLORS[i % COLORS.length].toString(16).padStart(6, "0")}"></span>
      <span class="indicator-name">${state.preset === "Solar" ? body.name : `body ${i + 1}`}</span>
      <span>m=${formatMass(body.mass)}</span>
      <span>v=${speed.toFixed(3)}</span>
      <span>az=${azimuth >= 0 ? "+" : ""}${azimuth.toFixed(1)}</span>
      <span>el=${elevation >= 0 ? "+" : ""}${elevation.toFixed(1)}</span>
    `;
    indicator.appendChild(row);
  }
}

function formatMass(mass) {
  if (Math.abs(mass) > 0 && Math.abs(mass) < 0.01) return mass.toExponential(2);
  if (Math.abs(mass) >= 1000) return mass.toExponential(2);
  return mass.toFixed(2);
}

function applyComFrame(bodies) {
  const totalMass = bodies.reduce((sum, b) => sum + b.mass, 0);
  const comPos = v();
  const comVel = v();
  for (const body of bodies) {
    comPos.addScaledVector(body.position, body.mass / totalMass);
    comVel.addScaledVector(body.velocity, body.mass / totalMass);
  }
  for (const body of bodies) {
    body.position.sub(comPos);
    body.velocity.sub(comVel);
  }
}

function rescaleToBound(bodies, config) {
  for (let n = 0; n < 200; n++) {
    if (totalEnergy(bodies, config) < 0) return;
    for (const body of bodies) body.velocity.multiplyScalar(0.975);
  }
}

function buildRandomBodies(config) {
  const rng = rand(config.seed);
  const n = Math.max(1, Math.min(8, config.bodyCount));
  const innerX = Math.max(0.5, config.limitX - 3);
  const innerY = Math.max(0.5, config.limitY - 3);
  const innerZ = Math.max(0.5, config.limitZ - 3);
  const inner = Math.min(innerX, innerY, innerZ);
  const minRadius = Math.max(0.2, 0.1 * inner);
  const minSep = Math.max(0.35, 0.16 * inner);
  const bodies = [];

  for (let i = 0; i < n; i++) {
    let position = v();
    for (let attempt = 0; attempt < 220; attempt++) {
      position = randomDirection(rng).multiplyScalar(minRadius + rng() * (0.65 * inner - minRadius));
      const separated = bodies.every((b) => b.position.distanceTo(position) > minSep);
      if (separated) break;
    }

    const mass = 4 + rng();
    const trial = randomDirection(rng);
    let tangent = new THREE.Vector3().crossVectors(position, trial);
    if (tangent.lengthSq() < 1e-10) tangent = new THREE.Vector3().crossVectors(position, v(1, 0, 0));
    tangent.normalize();
    const speed = (0.56 + rng() * 0.88) * (0.7 + rng() * 0.6);
    bodies.push({
      name: NAMES[i],
      mass,
      position,
      velocity: tangent.multiplyScalar(speed * 2.0),
      mesh: null,
      trail: null,
      points: [],
    });
  }

  applyComFrame(bodies);
  rescaleToBound(bodies, config);
  return bodies;
}

function buildSolarBodies(config) {
  const n = Math.max(1, Math.min(8, config.bodyCount));
  const inner = Math.max(0.5, Math.min(config.limitX, config.limitY, config.limitZ) - 3);
  const maxTarget = 0.95 * inner;
  const maxInput = Math.max(...SOLAR_RADII);
  const bodies = [];

  bodies.push({ name: SOLAR_NAMES[0], mass: SOLAR_MASSES[0], position: v(), velocity: v(), mesh: null, trail: null, points: [] });
  for (let i = 1; i < n; i++) {
    const rNorm = SOLAR_RADII[i] / maxInput;
    const r = maxTarget * Math.pow(rNorm, 0.65);
    const th = SOLAR_PHASES[i];
    const position = v(r * Math.cos(th), r * Math.sin(th), 0);
    const speed = Math.sqrt(Math.max(1e-9, config.gravity) * SOLAR_MASSES[0] / Math.max(r, 1e-9));
    const velocity = v(-speed * Math.sin(th), speed * Math.cos(th), 0);
    bodies.push({ name: SOLAR_NAMES[i], mass: SOLAR_MASSES[i], position, velocity, mesh: null, trail: null, points: [] });
  }
  applyComFrame(bodies);
  return bodies;
}

function accelerationAt(index, positions, bodies, config) {
  const acc = v();
  for (let j = 0; j < bodies.length; j++) {
    if (j === index) continue;
    const delta = positions[j].clone().sub(positions[index]);
    const dist2 = delta.lengthSq() + config.softening * config.softening;
    const invDist3 = 1 / (dist2 * Math.sqrt(dist2));
    acc.addScaledVector(delta, config.gravity * bodies[j].mass * invDist3);
  }
  return acc;
}

function rk4Step(bodies, config) {
  const dt = config.dt;
  const p0 = bodies.map((b) => b.position.clone());
  const v0 = bodies.map((b) => b.velocity.clone());
  const k1p = v0.map((x) => x.clone());
  const k1v = bodies.map((_, i) => accelerationAt(i, p0, bodies, config));

  const p2 = p0.map((p, i) => p.clone().addScaledVector(k1p[i], 0.5 * dt));
  const v2 = v0.map((vel, i) => vel.clone().addScaledVector(k1v[i], 0.5 * dt));
  const k2p = v2.map((x) => x.clone());
  const k2v = bodies.map((_, i) => accelerationAt(i, p2, bodies, config));

  const p3 = p0.map((p, i) => p.clone().addScaledVector(k2p[i], 0.5 * dt));
  const v3 = v0.map((vel, i) => vel.clone().addScaledVector(k2v[i], 0.5 * dt));
  const k3p = v3.map((x) => x.clone());
  const k3v = bodies.map((_, i) => accelerationAt(i, p3, bodies, config));

  const p4 = p0.map((p, i) => p.clone().addScaledVector(k3p[i], dt));
  const v4 = v0.map((vel, i) => vel.clone().addScaledVector(k3v[i], dt));
  const k4p = v4.map((x) => x.clone());
  const k4v = bodies.map((_, i) => accelerationAt(i, p4, bodies, config));

  for (let i = 0; i < bodies.length; i++) {
    bodies[i].position.copy(p0[i])
      .addScaledVector(k1p[i], dt / 6)
      .addScaledVector(k2p[i], dt / 3)
      .addScaledVector(k3p[i], dt / 3)
      .addScaledVector(k4p[i], dt / 6);
    bodies[i].velocity.copy(v0[i])
      .addScaledVector(k1v[i], dt / 6)
      .addScaledVector(k2v[i], dt / 3)
      .addScaledVector(k3v[i], dt / 3)
      .addScaledVector(k4v[i], dt / 6);
  }
}

function totalEnergy(bodies, config) {
  let kinetic = 0;
  let potential = 0;
  for (const body of bodies) kinetic += 0.5 * body.mass * body.velocity.lengthSq();
  for (let i = 0; i < bodies.length; i++) {
    for (let j = i + 1; j < bodies.length; j++) {
      const d = Math.sqrt(bodies[i].position.distanceToSquared(bodies[j].position) + config.softening * config.softening);
      potential -= config.gravity * bodies[i].mass * bodies[j].mass / d;
    }
  }
  return kinetic + potential;
}

function bodyRadius(body, bodies) {
  const mean = bodies.reduce((sum, b) => sum + b.mass, 0) / bodies.length;
  const base = Math.max(0.05, 0.16 - 0.01 * Math.max(0, bodies.length - 3));
  return Math.max(0.055, base * (0.85 + 0.3 * Math.pow(body.mass / (mean + 1e-12), 1 / 9)));
}

function clearGroup(group) {
  while (group.children.length) {
    const child = group.children.pop();
    child.geometry?.dispose();
    if (Array.isArray(child.material)) {
      child.material.forEach((material) => {
        material.map?.dispose();
        material.dispose();
      });
    } else {
      child.material?.map?.dispose();
      child.material?.dispose();
    }
  }
}

function niceStep(limit) {
  if (limit <= 6) return 2;
  if (limit <= 14) return 4;
  if (limit <= 24) return 6;
  return 10;
}

function roundRect(context, x, y, width, height, radius) {
  context.beginPath();
  context.moveTo(x + radius, y);
  context.lineTo(x + width - radius, y);
  context.quadraticCurveTo(x + width, y, x + width, y + radius);
  context.lineTo(x + width, y + height - radius);
  context.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
  context.lineTo(x + radius, y + height);
  context.quadraticCurveTo(x, y + height, x, y + height - radius);
  context.lineTo(x, y + radius);
  context.quadraticCurveTo(x, y, x + radius, y);
  context.closePath();
}

function textSprite(text, color = "#17201f") {
  const canvas = document.createElement("canvas");
  const context = canvas.getContext("2d");
  canvas.width = 256;
  canvas.height = 96;
  context.font = "600 34px Inter, Arial, sans-serif";
  context.textAlign = "center";
  context.textBaseline = "middle";
  context.fillStyle = "rgba(255,255,255,0.84)";
  context.strokeStyle = "rgba(23,32,31,0.18)";
  context.lineWidth = 5;
  roundRect(context, 18, 20, 220, 56, 14);
  context.fill();
  context.stroke();
  context.fillStyle = color;
  context.fillText(text, 128, 49);

  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  const material = new THREE.SpriteMaterial({ map: texture, transparent: true, depthTest: false });
  const sprite = new THREE.Sprite(material);
  sprite.scale.set(1.45, 0.54, 1);
  return sprite;
}

function addAxisLabels(axis, limit, step, color, config) {
  for (let value = -Math.floor(limit / step) * step; value <= limit + 1e-9; value += step) {
    if (Math.abs(value) < 1e-9) continue;
    const label = textSprite(String(value), color);
    if (axis === "x") label.position.set(value, -config.limitY, -config.limitZ);
    if (axis === "y") label.position.set(-config.limitX, value, -config.limitZ);
    if (axis === "z") label.position.set(-config.limitX, -config.limitY, value);
    helperGroup.add(label);
  }
}

function rebuildHelpers(config) {
  clearGroup(helperGroup);

  if (config.showGrid) {
    const material = new THREE.LineBasicMaterial({ color: 0xd8ded9, transparent: true, opacity: 0.32 });
    addGridFace("xy", config, material);
    addGridFace("xz", config, material);
    addGridFace("yz", config, material);
  }

  if (config.showAxisNumbers) {
    helperGroup.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([v(-config.limitX, -config.limitY, -config.limitZ), v(config.limitX, -config.limitY, -config.limitZ)]), new THREE.LineBasicMaterial({ color: 0xd64b54 })));
    helperGroup.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([v(-config.limitX, -config.limitY, -config.limitZ), v(-config.limitX, config.limitY, -config.limitZ)]), new THREE.LineBasicMaterial({ color: 0x2a9d8f })));
    helperGroup.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([v(-config.limitX, -config.limitY, -config.limitZ), v(-config.limitX, -config.limitY, config.limitZ)]), new THREE.LineBasicMaterial({ color: 0x1d5fb8 })));
    addAxisLabels("x", config.limitX, niceStep(config.limitX), "#a7353d", config);
    addAxisLabels("y", config.limitY, niceStep(config.limitY), "#1f786f", config);
    addAxisLabels("z", config.limitZ, niceStep(config.limitZ), "#244f9e", config);
  }
}

function addGridFace(face, config, material) {
  const points = [];
  const step = Math.max(1, niceStep(Math.max(config.limitX, config.limitY, config.limitZ)) / 2);

  if (face === "xy") {
    for (let x = -config.limitX; x <= config.limitX + 1e-9; x += step) points.push(v(x, -config.limitY, -config.limitZ), v(x, config.limitY, -config.limitZ));
    for (let y = -config.limitY; y <= config.limitY + 1e-9; y += step) points.push(v(-config.limitX, y, -config.limitZ), v(config.limitX, y, -config.limitZ));
  }
  if (face === "xz") {
    for (let x = -config.limitX; x <= config.limitX + 1e-9; x += step) points.push(v(x, -config.limitY, -config.limitZ), v(x, -config.limitY, config.limitZ));
    for (let z = -config.limitZ; z <= config.limitZ + 1e-9; z += step) points.push(v(-config.limitX, -config.limitY, z), v(config.limitX, -config.limitY, z));
  }
  if (face === "yz") {
    for (let y = -config.limitY; y <= config.limitY + 1e-9; y += step) points.push(v(-config.limitX, y, -config.limitZ), v(-config.limitX, y, config.limitZ));
    for (let z = -config.limitZ; z <= config.limitZ + 1e-9; z += step) points.push(v(-config.limitX, -config.limitY, z), v(-config.limitX, config.limitY, z));
  }

  helperGroup.add(new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(points), material.clone()));
}

function rebuildBounds(config) {
  clearGroup(boundsGroup);
  const boxGeometry = new THREE.BoxGeometry(config.limitX * 2, config.limitY * 2, config.limitZ * 2);
  const boxEdges = new THREE.EdgesGeometry(boxGeometry);
  boundsGroup.add(new THREE.LineSegments(boxEdges, new THREE.LineBasicMaterial({ color: 0x86938e, transparent: true, opacity: 0.55 })));
}

function rebuildScene() {
  clearGroup(bodyGroup);
  clearGroup(trailGroup);
  clearGroup(helperGroup);

  const config = readConfig();
  rebuildHelpers(config);
  rebuildBounds(config);
  state.bodies = state.preset === "Solar" ? buildSolarBodies(config) : buildRandomBodies(config);
  state.step = 0;
  state.time = 0;
  state.escaped = false;
  state.stopReason = "";
  state.running = true;
  document.getElementById("toggle").textContent = "Pause";

  for (let i = 0; i < state.bodies.length; i++) {
    const body = state.bodies[i];
    const color = COLORS[i % COLORS.length];
    const radius = bodyRadius(body, state.bodies);
    const mesh = new THREE.Mesh(
      new THREE.SphereGeometry(radius, 28, 18),
      new THREE.MeshStandardMaterial({ color, roughness: 0.45, metalness: 0.08 })
    );
    const trail = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints([body.position.clone()]),
      new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.8 })
    );
    body.mesh = mesh;
    body.trail = trail;
    body.points = [body.position.clone()];
    bodyGroup.add(mesh);
    trailGroup.add(trail);
  }

  updateVisuals();
  updateLabels();
}

function checkBoundary(config) {
  for (const body of state.bodies) {
    if (Math.abs(body.position.x) > config.limitX || Math.abs(body.position.y) > config.limitY || Math.abs(body.position.z) > config.limitZ) {
      state.escaped = true;
      state.stopReason = `body ${state.bodies.indexOf(body) + 1} reached the boundary`;
      state.running = false;
      document.getElementById("toggle").textContent = "Run";
      return;
    }
  }
}

function checkCollision() {
  for (let i = 0; i < state.bodies.length; i++) {
    for (let j = i + 1; j < state.bodies.length; j++) {
      const threshold = (bodyRadius(state.bodies[i], state.bodies) + bodyRadius(state.bodies[j], state.bodies)) * 0.2;
      if (state.bodies[i].position.distanceTo(state.bodies[j].position) <= threshold) {
        state.escaped = true;
        state.stopReason = `body ${i + 1} collided with body ${j + 1}`;
        state.running = false;
        document.getElementById("toggle").textContent = "Run";
        return;
      }
    }
  }
}

function updateVisuals() {
  const config = readConfig();
  for (const body of state.bodies) {
    body.mesh.position.copy(body.position);
    body.points.push(body.position.clone());
    if (body.points.length > config.trailLength) body.points.splice(0, body.points.length - config.trailLength);
    body.trail.geometry.dispose();
    body.trail.geometry = new THREE.BufferGeometry().setFromPoints(body.points);
  }
}

function tick() {
  const config = readConfig();
  if (state.running && !state.escaped) {
    for (let i = 0; i < config.stepsPerFrame; i++) {
      rk4Step(state.bodies, config);
      state.step += 1;
      state.time += config.dt;
      checkBoundary(config);
      checkCollision();
      if (!state.running) break;
    }
    updateVisuals();
    updateLabels();
  }

  if (controls.noPan) controls.target.set(0, 0, 0);
  controls.update();
  renderer.render(scene, camera);
  requestAnimationFrame(tick);
}

function resize() {
  const rect = viewport.getBoundingClientRect();
  renderer.setSize(rect.width, rect.height, false);
  camera.aspect = rect.width / Math.max(1, rect.height);
  camera.updateProjectionMatrix();
  configureControls();
  controls.handleResize();
}

window.addEventListener("resize", resize);
window.addEventListener("orientationchange", configureControls);

for (const input of Object.values(ui)) {
  input.addEventListener("input", updateLabels);
}

ui.bodyCount.addEventListener("change", () => {
  if (state.preset === "Solar") {
    state.preset = "Random";
    document.getElementById("preset-random").classList.add("active");
    document.getElementById("preset-solar").classList.remove("active");
    state.seedBump += 1;
  }
  rebuildScene();
});

ui.gravity.addEventListener("change", rebuildScene);

for (const id of ["limitX", "limitY", "limitZ"]) {
  ui[id].addEventListener("change", () => {
    const config = readConfig();
    rebuildBounds(config);
    rebuildHelpers(config);
    updateLabels();
  });
}

for (const id of ["showGrid", "showAxisNumbers"]) {
  ui[id].addEventListener("change", () => rebuildHelpers(readConfig()));
}

document.getElementById("preset-random").addEventListener("click", () => {
  state.preset = "Random";
  state.seedBump += 1;
  document.getElementById("preset-random").classList.add("active");
  document.getElementById("preset-solar").classList.remove("active");
  ui.bodyCount.value = 3;
  rebuildScene();
});

document.getElementById("preset-solar").addEventListener("click", () => {
  state.preset = "Solar";
  document.getElementById("preset-solar").classList.add("active");
  document.getElementById("preset-random").classList.remove("active");
  ui.bodyCount.value = 8;
  rebuildScene();
});

document.getElementById("toggle").addEventListener("click", () => {
  if (state.escaped) return;
  state.running = !state.running;
  document.getElementById("toggle").textContent = state.running ? "Pause" : "Run";
  updateLabels();
});

document.getElementById("restart").addEventListener("click", rebuildScene);
document.getElementById("randomize").addEventListener("click", () => {
  state.seedBump += 1;
  rebuildScene();
});

resize();
rebuildScene();
requestAnimationFrame(tick);
</script>

<style>
  * { box-sizing: border-box; }
  html, body { margin: 0; background: #f4f0e8; color: #17201f; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
  #app { min-height: 100vh; display: grid; grid-template-columns: 300px minmax(0, 1fr); }
  .panel { background: #17201f; color: #f8f3ea; padding: 22px 18px; overflow-y: auto; max-height: 100vh; }
  h1 { margin: 0 0 22px; font-size: 30px; line-height: 1; letter-spacing: 0; }
  label { display: block; margin: 16px 0 8px; color: #c4cfc8; font-size: 12px; text-transform: uppercase; letter-spacing: 0.06em; }
  input { width: 100%; height: 34px; border: 1px solid #53625d; background: #24302d; color: #f8f3ea; border-radius: 6px; padding: 0 10px; color-scheme: dark; }
  input[type="range"] { padding: 0; accent-color: #79b8ae; }
  .triple { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; }
  .switches { display: grid; grid-template-columns: 1fr; gap: 8px; margin-top: 14px; }
  .check { margin: 0; display: flex; align-items: center; gap: 8px; color: #f8f3ea; font-size: 13px; text-transform: none; letter-spacing: 0; }
  .check input { width: 16px; height: 16px; accent-color: #79b8ae; }
  .segmented, .actions { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
  button { height: 38px; border: 0; border-radius: 6px; background: #2a6f66; color: white; font-weight: 700; cursor: pointer; }
  button:hover { background: #245e57; }
  button.active { background: #79b8ae; color: #10201d; }
  .actions { margin-top: 20px; }
  .wide { width: 100%; margin-top: 10px; background: #725c2a; }
  .stage { min-width: 0; padding: 18px; }
  .metrics { display: grid; grid-template-columns: minmax(280px, 2fr) repeat(3, minmax(110px, 1fr)); gap: 12px; margin-bottom: 14px; }
  .metric { background: white; border: 1px solid #ded6ca; border-radius: 8px; padding: 12px 14px; min-height: 70px; }
  .metric span { color: #64706b; font-size: 12px; text-transform: uppercase; letter-spacing: 0.06em; display: block; }
  .metric strong { display: block; margin-top: 8px; font-size: 22px; white-space: normal; overflow: visible; overflow-wrap: anywhere; line-height: 1.15; }
  .status-card strong { font-size: 20px; }
  #viewport { position: relative; height: calc(100vh - 112px); background: white; border: 1px solid #ded6ca; border-radius: 8px; overflow: hidden; touch-action: none; }
  canvas { display: block; width: 100%; height: 100%; }
  #indicator { position: absolute; left: 12px; bottom: 12px; max-width: min(520px, calc(100% - 24px)); padding: 10px 12px; border: 1px solid rgba(23,32,31,0.16); border-radius: 8px; background: rgba(255,255,255,0.86); backdrop-filter: blur(8px); color: #17201f; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12px; line-height: 1.35; pointer-events: none; }
  .indicator-row { display: grid; grid-template-columns: 14px 58px repeat(4, auto); gap: 9px; align-items: center; white-space: nowrap; }
  .indicator-row + .indicator-row { margin-top: 4px; }
  .indicator-dot { width: 10px; height: 10px; border-radius: 999px; display: inline-block; box-shadow: 0 0 0 1px rgba(23,32,31,0.18); }
  .indicator-name { font-weight: 700; font-family: Inter, ui-sans-serif, system-ui, sans-serif; }
  @media (max-width: 860px) {
    #app { grid-template-columns: 1fr; }
    .panel { max-height: none; }
    .metrics { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .status-card { grid-column: 1 / -1; }
    #viewport { height: 68vh; }
  }
</style>
"""


def main() -> None:
    st.set_page_config(page_title="Three-Body Simulation", page_icon="3", layout="wide")
    st.markdown(
        """
        <style>
          #MainMenu, header, footer, [data-testid="stToolbar"], [data-testid="stDecoration"],
          [data-testid="stStatusWidget"], .stDeployButton {
            display: none !important;
          }
          .block-container {
            padding-top: 0 !important;
            padding-bottom: 0 !important;
            max-width: 100% !important;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.components.v1.html(APP_HTML, height=920, scrolling=True)


if __name__ == "__main__":
    main()
