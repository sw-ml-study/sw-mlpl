import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

let scene, camera, renderer, controls, animId;

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

window.__stage3d_add_step = function(json) {
    if (!scene) return;
    const ev = typeof json === 'string' ? JSON.parse(json) : json;
    const x = ev.step_idx * SPACING;
    const color = opColor(ev.label);
    const box = new THREE.Mesh(
        new THREE.BoxGeometry(1.2, 0.8, 0.8),
        new THREE.MeshStandardMaterial({ color })
    );
    box.position.set(x, 0.5, 0);
    box.castShadow = true;
    scene.add(box);

    const label = makeLabel(ev.label);
    label.position.set(x, 1.4, 0);
    scene.add(label);

    camera.position.set(x + 3, 4, 8);
    controls.target.set(x, 0.5, 0);
    controls.update();
};
