import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

let scene, camera, renderer, controls, animId;
let stepCount = 0;
let viewStep = 0;
const stepObjects = [];

function createGround() {
    const grid = new THREE.GridHelper(40, 40, 0x444466, 0x222233);
    grid.position.y = 0;
    scene.add(grid);

    const plane = new THREE.Mesh(
        new THREE.PlaneGeometry(40, 40),
        new THREE.MeshStandardMaterial({ color: 0x181825, roughness: 0.9 })
    );
    plane.rotation.x = -Math.PI / 2;
    plane.position.y = -0.01;
    plane.receiveShadow = true;
    scene.add(plane);
}

function createLights() {
    scene.add(new THREE.AmbientLight(0x8888aa, 0.6));
    const dir = new THREE.DirectionalLight(0xffffff, 0.8);
    dir.position.set(5, 10, 7);
    dir.castShadow = true;
    scene.add(dir);
}

function animate() {
    animId = requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
}

function resize() {
    const canvas = renderer.domElement;
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    if (canvas.width !== w || canvas.height !== h) {
        renderer.setSize(w, h, false);
        camera.aspect = w / h;
        camera.updateProjectionMatrix();
    }
}

window.__stage3d_init = function(canvas) {
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0a0a14);

    camera = new THREE.PerspectiveCamera(50, canvas.clientWidth / canvas.clientHeight, 0.1, 200);
    camera.position.set(0, 6, 12);

    renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(canvas.clientWidth, canvas.clientHeight, false);
    renderer.shadowMap.enabled = true;

    controls = new OrbitControls(camera, canvas);
    controls.target.set(0, 0.5, 0);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.update();

    createGround();
    createLights();

    window.addEventListener('resize', resize);
    animate();
};

window.__stage3d_destroy = function() {
    if (animId) cancelAnimationFrame(animId);
    window.removeEventListener('resize', resize);
    if (renderer) renderer.dispose();
    if (controls) controls.dispose();
    scene = camera = renderer = controls = animId = null;
};

const SPACING = 2.5;
const OP_COLORS = {
    matmul: 0x6688cc, train: 0xcc6666, softmax: 0x66cc88,
    cross_entropy: 0xcc6666, reshape: 0x88aacc, default: 0x8888aa
};

function opColor(label) {
    for (const [key, color] of Object.entries(OP_COLORS)) {
        if (label.includes(key)) return color;
    }
    return OP_COLORS.default;
}

function makeLabel(text) {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    canvas.width = 256; canvas.height = 64;
    ctx.fillStyle = '#1e1e2e'; ctx.fillRect(0, 0, 256, 64);
    ctx.fillStyle = '#cdd6f4'; ctx.font = '14px monospace';
    ctx.fillText(text.substring(0, 36), 8, 24);
    const tex = new THREE.CanvasTexture(canvas);
    const mat = new THREE.SpriteMaterial({ map: tex });
    const sprite = new THREE.Sprite(mat);
    sprite.scale.set(2.5, 0.6, 1);
    return sprite;
}

function shapeMesh(shape, color) {
    const rank = shape.length;
    const mat = new THREE.MeshStandardMaterial({ color, transparent: true, opacity: 0.85 });
    if (rank === 0) {
        return new THREE.Mesh(new THREE.SphereGeometry(0.3, 16, 16), mat);
    }
    if (rank === 1) {
        const w = Math.min(shape[0] * 0.15, 3);
        return new THREE.Mesh(new THREE.BoxGeometry(w, 0.25, 0.25), mat);
    }
    if (rank === 2) {
        const w = Math.min(shape[1] * 0.15, 3);
        const h = Math.min(shape[0] * 0.15, 2);
        return new THREE.Mesh(new THREE.BoxGeometry(w, h, 0.12), mat);
    }
    const w = Math.min((shape[2] || 1) * 0.12, 2.5);
    const h = Math.min((shape[1] || 1) * 0.12, 1.5);
    const group = new THREE.Group();
    const layers = Math.min(shape[0] || 1, 8);
    for (let i = 0; i < layers; i++) {
        const slab = new THREE.Mesh(new THREE.BoxGeometry(w, h, 0.08), mat.clone());
        slab.position.z = i * 0.15 - (layers * 0.15) / 2;
        slab.castShadow = true;
        group.add(slab);
    }
    return group;
}

window.__stage3d_add_step = function(ev) {
    if (!scene || !ev) return;
    const x = stepCount * SPACING;
    stepCount++;
    const color = opColor(ev.label);
    const shape = ev.output?.shape || [];
    const mesh = shapeMesh(shape, color);
    mesh.position.set(x, 0.6, 0);
    if (mesh.castShadow !== undefined) mesh.castShadow = true;
    scene.add(mesh);
    stepObjects.push(mesh);

    const rank = shape.length;
    const dims = shape.length ? `[${shape.join(',')}]` : 'scalar';
    const text = `${ev.output?.name || ''}: ${dims} R${rank}`;
    const label = makeLabel(text);
    label.position.set(x, 1.6, 0);
    scene.add(label);
    stepObjects.push(label);

    viewStep = stepCount - 1;
    camera.position.set(x + 3, 4, 8);
    controls.target.set(x, 0.5, 0);
    controls.update();
};

function panToStep(idx) {
    if (!camera || idx < 0 || idx >= stepCount) return;
    viewStep = idx;
    const x = idx * SPACING;
    camera.position.set(x + 3, 4, 8);
    controls.target.set(x, 0.5, 0);
    controls.update();
}

window.__stage3d_prev = function() { panToStep(viewStep - 1); };
window.__stage3d_next = function() { panToStep(viewStep + 1); };
window.__stage3d_home = function() { panToStep(0); };
window.__stage3d_end = function() { panToStep(stepCount - 1); };

window.__stage3d_clear = function() {
    for (const obj of stepObjects) scene.remove(obj);
    stepObjects.length = 0;
    stepCount = 0;
    viewStep = 0;
    if (camera) {
        camera.position.set(0, 6, 12);
        controls.target.set(0, 0.5, 0);
        controls.update();
    }
};
