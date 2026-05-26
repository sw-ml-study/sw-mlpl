import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

let scene, camera, renderer, controls, animId;
let stepCount = 0;
let viewStep = 0;
const stepObjects = [];
const activeAnims = [];
const pendingEvents = [];
let prevMesh = null;

let groundPlane, groundGrid;

function createGround() {
    groundGrid = new THREE.GridHelper(2000, 2000, 0xcccccc, 0xdddddd);
    groundGrid.position.y = 0;
    scene.add(groundGrid);

    groundPlane = new THREE.Mesh(
        new THREE.PlaneGeometry(2000, 200),
        new THREE.MeshStandardMaterial({ color: 0xeeeee8, roughness: 0.95 })
    );
    groundPlane.rotation.x = -Math.PI / 2;
    groundPlane.position.y = -0.01;
    groundPlane.receiveShadow = true;
    scene.add(groundPlane);
}

function createMountains() {
    const farMat = new THREE.MeshStandardMaterial({ color: 0x8B6B4A, flatShading: true });
    const nearMat = new THREE.MeshStandardMaterial({ color: 0x5A8A4A, flatShading: true });
    for (let i = -40; i < 40; i++) {
        const x = i * 20 + Math.random() * 8;
        const h = 8 + Math.random() * 12;
        const w = 6 + Math.random() * 8;
        const far = new THREE.Mesh(new THREE.ConeGeometry(w, h, 5 + Math.floor(Math.random() * 3)), farMat);
        far.position.set(x, h / 2, -60 - Math.random() * 20);
        scene.add(far);
    }
    for (let i = -40; i < 40; i++) {
        const x = i * 15 + Math.random() * 6;
        const h = 3 + Math.random() * 5;
        const w = 4 + Math.random() * 5;
        const near = new THREE.Mesh(new THREE.ConeGeometry(w, h, 4 + Math.floor(Math.random() * 3)), nearMat);
        near.position.set(x, h / 2, -25 - Math.random() * 15);
        scene.add(near);
    }
}

function createLights() {
    scene.add(new THREE.AmbientLight(0xffffff, 0.7));
    const dir = new THREE.DirectionalLight(0xfff8e8, 0.9);
    dir.position.set(10, 15, 10);
    dir.castShadow = true;
    scene.add(dir);
    const fill = new THREE.DirectionalLight(0x8888cc, 0.3);
    fill.position.set(-5, 5, 10);
    scene.add(fill);
}

function animate() {
    animId = requestAnimationFrame(animate);
    const now = performance.now();
    for (let i = activeAnims.length - 1; i >= 0; i--) {
        const a = activeAnims[i];
        const t = Math.min((now - a.start) / a.duration, 1);
        a.update(t);
        if (t >= 1) activeAnims.splice(i, 1);
    }
    controls.update();
    renderer.render(scene, camera);
}

function easeOutBack(t) {
    const c = 1.7;
    return 1 + (t - 1) ** 3 * (c + 1) + (t - 1) ** 2 * c;
}

function animateEntry(mesh) {
    mesh.scale.set(0, 0, 0);
    activeAnims.push({
        start: performance.now(),
        duration: 500,
        update(t) {
            const s = easeOutBack(t);
            mesh.scale.set(s, s, s);
        }
    });
}

function dimPrevious(mesh) {
    if (!mesh) return;
    const mats = mesh.material ? [mesh.material] : [];
    if (mesh.children) mesh.children.forEach(c => { if (c.material) mats.push(c.material); });
    for (const m of mats) {
        m.emissive?.setHex(0x000000);
        m.opacity = 0.6;
    }
}

function glowNew(mesh, color) {
    const mats = mesh.material ? [mesh.material] : [];
    if (mesh.children) mesh.children.forEach(c => { if (c.material) mats.push(c.material); });
    for (const m of mats) m.emissive?.setHex(color);
    activeAnims.push({
        start: performance.now(),
        duration: 1000,
        update(t) {
            const intensity = 1 - t;
            for (const m of mats) {
                if (m.emissiveIntensity !== undefined) m.emissiveIntensity = intensity * 0.5;
            }
        }
    });
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
    if (animId) cancelAnimationFrame(animId);
    if (controls) controls.dispose();
    if (renderer) renderer.dispose();

    if (!scene) {
        scene = new THREE.Scene();
        scene.background = new THREE.Color(0xd8e4f0);
        scene.fog = new THREE.FogExp2(0xd8e4f0, 0.008);
        createGround();
        createMountains();
        createLights();
    }
    if (!camera) {
        camera = new THREE.PerspectiveCamera(50, canvas.clientWidth / canvas.clientHeight, 0.1, 200);
        camera.position.set(0, 6, 12);
    }

    renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(canvas.clientWidth, canvas.clientHeight, false);
    renderer.shadowMap.enabled = true;

    controls = new OrbitControls(camera, canvas);
    controls.target.set(controls.target.x, 0.5, 0);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.update();

    canvas.tabIndex = 0;
    canvas.addEventListener('keydown', onCanvasKey);
    canvas.addEventListener('click', onCanvasClick);
    window.addEventListener('resize', resize);
    animate();

    if (pendingEvents.length > 0) {
        const queued = pendingEvents.splice(0);
        for (const ev of queued) window.__stage3d_add_step(ev);
    }
};

const raycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2();
let selectedMesh = null;

function onCanvasClick(e) {
    if (!renderer || !camera) return;
    const rect = renderer.domElement.getBoundingClientRect();
    mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
    raycaster.setFromCamera(mouse, camera);
    const meshes = stepObjects.filter(o => o.isMesh || o.isGroup);
    const targets = [];
    for (const obj of meshes) {
        if (obj.isMesh) targets.push(obj);
        if (obj.children) obj.children.forEach(c => { if (c.isMesh) targets.push(c); });
    }
    const hits = raycaster.intersectObjects(targets, false);
    if (hits.length === 0) return;
    const hit = hits[0].object;
    const ud = hit.userData;
    if (!ud) return;
    selectMesh(hit);
    showDetail(ud);
}

let selectionPointer = null;

function selectMesh(mesh) {
    if (selectionPointer) { scene.remove(selectionPointer); selectionPointer = null; }
    const pos = mesh.position?.clone() || mesh.parent?.position?.clone();
    if (!pos) return;
    const geo = new THREE.ConeGeometry(0.15, 0.4, 4);
    geo.rotateX(Math.PI);
    const mat = new THREE.MeshStandardMaterial({ color: 0xffcc44, emissive: 0xffcc44, emissiveIntensity: 0.6 });
    selectionPointer = new THREE.Mesh(geo, mat);
    selectionPointer.position.set(pos.x, pos.y + 1.2, pos.z);
    scene.add(selectionPointer);
}

function showDetail(ud) {
    const el = document.getElementById('stage3d-detail');
    if (!el) return;
    const name = ud.varName || '';
    const label = ud.label || '';
    const shape = ud.shape || [];
    const rank = shape.length;
    const dims = shape.length ? `[${shape.join(', ')}]` : 'scalar';
    const elements = ud.elements || (shape.length ? shape.reduce((a, b) => a * b, 1) : 1);
    const bytes = elements * 8;
    const mem = bytes < 1024 ? `${bytes} B` : bytes < 1048576 ? `${(bytes / 1024).toFixed(1)} KB` : `${(bytes / 1048576).toFixed(1)} MB`;
    el.innerHTML = `
        <div class="stage3d-detail-title">${name ? name + ' =' : ''} ${label}</div>
        <div class="stage3d-detail-row"><strong>Shape:</strong> ${dims} &nbsp; <strong>Rank:</strong> ${rank} &nbsp; <strong>Elements:</strong> ${elements.toLocaleString()} &nbsp; <strong>Memory:</strong> ~${mem}</div>
        <div class="stage3d-detail-row"><strong>Step:</strong> ${ud.stepIdx !== undefined ? ud.stepIdx + 1 : '?'} of ${stepCount}</div>
    `;
    el.style.display = 'block';
}

function onCanvasKey(e) {
    if (e.key === 'ArrowRight') { e.preventDefault(); panToStep(viewStep + 1); }
    else if (e.key === 'ArrowLeft') { e.preventDefault(); panToStep(viewStep - 1); }
    else if (e.key === 'Home') { e.preventDefault(); panToStep(0); }
    else if (e.key === 'End') { e.preventDefault(); panToStep(stepCount - 1); }
}

window.__stage3d_destroy = function() {
    // No-op: scene persists across Yew component remounts.
    // Only :3d off / :clear destroys state.
};

window.__stage3d_teardown = function() {
    if (animId) cancelAnimationFrame(animId);
    window.removeEventListener('resize', resize);
    if (renderer) renderer.dispose();
    if (controls) controls.dispose();
    scene = camera = renderer = controls = animId = null;
};

const SPACING = 2.5;
const OP_COLORS = {
    matmul: 0x44aaff, train: 0xff5577, softmax: 0x44ffaa,
    cross_entropy: 0xff6688, reshape: 0xaa88ff, reduce: 0xffaa44,
    iota: 0x00ddcc, randn: 0xff88dd, svg: 0xffcc44, default: 0x66bbff
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
    canvas.width = 320; canvas.height = 48;
    ctx.fillStyle = 'rgba(20, 30, 60, 0.65)';
    ctx.roundRect(2, 2, 316, 44, 12); ctx.fill();
    ctx.fillStyle = '#ffffff'; ctx.font = 'bold 15px monospace';
    ctx.fillText(text.substring(0, 40), 12, 30);
    const tex = new THREE.CanvasTexture(canvas);
    const mat = new THREE.SpriteMaterial({ map: tex, transparent: true });
    const sprite = new THREE.Sprite(mat);
    sprite.scale.set(3, 0.45, 1);
    return sprite;
}

function shapeMesh(shape, color) {
    const rank = shape.length;
    const mat = new THREE.MeshStandardMaterial({ color, roughness: 0.3, metalness: 0.1 });
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
    if (!ev) return;
    if (!scene) {
        pendingEvents.push(ev);
        if (pendingEvents.length === 1) {
            const poll = () => {
                if (scene && pendingEvents.length > 0) {
                    const queued = pendingEvents.splice(0);
                    for (const e of queued) window.__stage3d_add_step(e);
                } else if (pendingEvents.length > 0) {
                    requestAnimationFrame(poll);
                }
            };
            requestAnimationFrame(poll);
        }
        return;
    }
    const x = stepCount * SPACING;
    stepCount++;
    const color = opColor(ev.label);
    const shape = ev.output?.shape || [];
    dimPrevious(prevMesh);
    const mesh = shapeMesh(shape, color);
    mesh.position.set(x, 0.6, 0);
    if (mesh.castShadow !== undefined) mesh.castShadow = true;
    const rawName = ev.output?.name || '';
    const isVar = ev.label.includes('=');
    const varName = isVar ? rawName : null;
    mesh.userData = { varName, label: ev.label, stepIdx: stepCount - 1, shape, elements: ev.output?.elements || 0 };
    if (mesh.children) mesh.children.forEach(c => c.userData = mesh.userData);
    scene.add(mesh);
    stepObjects.push(mesh);
    prevMesh = mesh;

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

function clearSelection() {
    if (selectionPointer) { scene.remove(selectionPointer); selectionPointer = null; }
    const el = document.getElementById('stage3d-detail');
    if (el) el.style.display = 'none';
}

function panToStep(idx) {
    if (!camera || idx < 0 || idx >= stepCount) return;
    const dx = (idx - viewStep) * SPACING;
    viewStep = idx;
    camera.position.x += dx;
    controls.target.x += dx;
    controls.update();
}

window.__stage3d_prev = function() { clearSelection(); panToStep(viewStep - 1); };
window.__stage3d_next = function() { clearSelection(); panToStep(viewStep + 1); };
window.__stage3d_home = function() { clearSelection(); panToStep(0); };
window.__stage3d_end = function() { clearSelection(); panToStep(stepCount - 1); };

window.__stage3d_clear = function() {
    for (const obj of stepObjects) scene.remove(obj);
    stepObjects.length = 0;
    activeAnims.length = 0;
    pendingEvents.length = 0;
    stepCount = 0;
    viewStep = 0;
    prevMesh = null;
    if (camera) {
        camera.position.set(0, 6, 12);
        controls.target.set(0, 0.5, 0);
        controls.update();
    }
};
