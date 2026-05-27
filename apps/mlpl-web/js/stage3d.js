import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

let scene, camera, renderer, controls, animId;
let stepCount = 0;
let viewStep = 0;
let selectedStepIdx = -1;
const stepObjects = [];
const stepMeshes = [];
const activeAnims = [];
const pendingEvents = [];
const varPositions = {};
let prevMesh = null;

function createBackdrop() {
    new THREE.TextureLoader().load('salt-flat.jpg', tex => {
        const aspect = tex.image.width / tex.image.height;
        const h = 120;
        const w = h * aspect * 6;
        const mat = new THREE.MeshBasicMaterial({ map: tex, fog: false });
        const plane = new THREE.Mesh(new THREE.PlaneGeometry(w, h), mat);
        plane.position.set(0, h * 0.17, -140);
        scene.add(plane);
    });
}

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

function createLegend() {
    const marks = [10, 100, 1000];
    const labels = ['10 elements', '100 elements', '1K elements'];
    const mat = new THREE.MeshStandardMaterial({ color: 0x8899bb, roughness: 0.3, metalness: 0.1 });
    const midX = -2 * SPACING;
    for (let i = 0; i < marks.length; i++) {
        const d = logDim(marks[i]);
        const cube = new THREE.Mesh(new THREE.BoxGeometry(0.4, 0.15, d), mat.clone());
        const x = -(i + 1) * SPACING;
        cube.position.set(x, 0.08, -d / 2);
        scene.add(cube);
        const lbl = makeLabel(labels[i], { fontSize: 18, color: '#aabbdd', bg: 'rgba(20, 30, 60, 0.7)', w: 280, h: 44, scale: [2.5, 0.4, 1] });
        lbl.position.set(x, 0.6, 0);
        scene.add(lbl);
    }
    const title = makeLabel('LOGARITHMIC SCALE', { fontSize: 20, color: '#ffcc44', bg: 'rgba(20, 30, 60, 0.8)', w: 380, h: 48, scale: [3.5, 0.5, 1] });
    title.position.set(midX, 1.2, 0);
    scene.add(title);
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
        near.position.set(x, h / 2, -45 - Math.random() * 12);
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
        scene.background = new THREE.Color(0xc8d8e8);
        scene.fog = new THREE.FogExp2(0xc8d8e8, 0.004);
        createBackdrop();
        createGround();
        createLegend();
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
    const worldPos = new THREE.Vector3();
    const target = mesh.parent?.isGroup ? mesh.parent : mesh;
    target.getWorldPosition(worldPos);
    const geo = new THREE.ConeGeometry(0.15, 0.4, 4);
    geo.rotateX(Math.PI);
    const mat = new THREE.MeshStandardMaterial({ color: 0xffcc44, emissive: 0xffcc44, emissiveIntensity: 0.6 });
    selectionPointer = new THREE.Mesh(geo, mat);
    selectionPointer.position.set(worldPos.x, worldPos.y + 1.5, worldPos.z);
    scene.add(selectionPointer);
    selectedStepIdx = mesh.userData?.stepIdx ?? -1;
}

function selectStep(idx) {
    if (idx < 0 || idx >= stepMeshes.length) return;
    const mesh = stepMeshes[idx];
    if (!mesh) return;
    selectMesh(mesh);
    showDetail(mesh.userData);
    panToStep(idx);
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
    let html = `
        <div class="stage3d-detail-title">${name ? name + ' =' : ''} ${label}</div>
        <div class="stage3d-detail-row"><strong>Shape:</strong> ${dims} &nbsp; <strong>Rank:</strong> ${rank} &nbsp; <strong>Elements:</strong> ${elements.toLocaleString()} &nbsp; <strong>Memory:</strong> ~${mem}</div>
        <div class="stage3d-detail-row"><strong>Step:</strong> ${ud.stepIdx !== undefined ? ud.stepIdx + 1 : '?'} of ${stepCount}</div>`;
    const vals = ud.values;
    const summary = ud.summary;
    if (elements <= 8 && rank === 1 && elements > 0) {
        html += `<div class="stage3d-detail-row" style="color:var(--peach)"><strong>Compact representation</strong> (${elements} elements${elements <= 4 ? ' -- bottleneck?' : ''})</div>`;
    }
    if (vals && vals.length > 0) {
        const stats = computeStats(vals);
        html += renderStats(stats);
        html += renderValues(vals, shape);
    } else if (summary) {
        html += renderStats(summary);
        html += renderHistogram(summary.histogram);
    }
    el.innerHTML = html;
    el.style.display = 'block';
}

function computeStats(vals) {
    const min = Math.min(...vals), max = Math.max(...vals);
    const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
    const std = Math.sqrt(vals.reduce((a, v) => a + (v - mean) ** 2, 0) / vals.length);
    return { min, max, mean, std };
}

function renderStats(s) {
    return `<div class="stage3d-detail-row"><strong>Min:</strong> ${s.min.toFixed(4)} &nbsp; <strong>Max:</strong> ${s.max.toFixed(4)} &nbsp; <strong>Mean:</strong> ${s.mean.toFixed(4)} &nbsp; <strong>Std:</strong> ${s.std.toFixed(4)}</div>`;
}

function renderValues(vals, shape) {
    const show = vals.slice(0, 20);
    const fmt = show.map(v => v % 1 === 0 ? v.toString() : v.toFixed(3));
    const suffix = vals.length > 20 ? ` ... (${vals.length - 20} more)` : '';
    if (shape.length === 2 && shape[1] <= 10 && vals.length <= 20) {
        let table = '<div class="stage3d-detail-row" style="font-family:var(--mono);font-size:12px">';
        for (let r = 0; r < shape[0] && r * shape[1] < 20; r++) {
            const row = fmt.slice(r * shape[1], (r + 1) * shape[1]).join('  ');
            table += row + '<br>';
        }
        return table + '</div>';
    }
    return `<div class="stage3d-detail-row" style="font-family:var(--mono);font-size:12px">${fmt.join(', ')}${suffix}</div>`;
}

function renderHistogram(hist) {
    if (!hist || hist.length === 0) return '';
    const max = Math.max(...hist);
    const bars = hist.map(h => {
        const pct = max > 0 ? (h / max * 100) : 0;
        return `<span style="display:inline-block;width:12px;height:${pct * 0.3}px;background:#ffaa44;margin:0 1px;vertical-align:bottom"></span>`;
    }).join('');
    return `<div class="stage3d-detail-row" style="height:35px;line-height:35px">${bars}</div>`;
}

function onCanvasKey(e) {
    if (e.key === 'ArrowRight') {
        e.preventDefault();
        if (selectedStepIdx >= 0) selectStep(selectedStepIdx + 1); else panToStep(viewStep + 1);
    } else if (e.key === 'ArrowLeft') {
        e.preventDefault();
        if (selectedStepIdx >= 0) selectStep(selectedStepIdx - 1); else panToStep(viewStep - 1);
    } else if (e.key === 'Home') { e.preventDefault(); clearSelection(); panToStep(0); }
    else if (e.key === 'End') { e.preventDefault(); clearSelection(); panToStep(stepCount - 1); }
    else if (e.key === 'Escape') { clearSelection(); }
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

const SPACING = 5.0;
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

function makeLabel(text, opts) {
    const { fontSize = 18, color = '#ffffff', bg = 'rgba(20, 30, 60, 0.7)', w = 400, h = 56, scale = [3.5, 0.5, 1] } = opts || {};
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    canvas.width = w; canvas.height = h;
    ctx.fillStyle = bg;
    ctx.roundRect(2, 2, w - 4, h - 4, 12); ctx.fill();
    ctx.fillStyle = color; ctx.font = `bold ${fontSize}px monospace`;
    ctx.fillText(text.substring(0, 50), 14, h * 0.6);
    const tex = new THREE.CanvasTexture(canvas);
    const mat = new THREE.SpriteMaterial({ map: tex, transparent: true });
    const sprite = new THREE.Sprite(mat);
    sprite.scale.set(...scale);
    return sprite;
}

function logDim(n) {
    return Math.max(0.3, Math.min(6, Math.log2(Math.max(n, 1)) * 0.5 + 0.3));
}

function valueColor(v, min, max) {
    const range = Math.abs(max - min) < 1e-12 ? 1 : max - min;
    const t = (v - min) / range;
    const r = t < 0.5 ? 0.3 : 0.3 + (t - 0.5) * 2 * 0.7;
    const g = 1 - Math.abs(t - 0.5) * 2;
    const b = t > 0.5 ? 0.3 : 0.3 + (0.5 - t) * 2 * 0.7;
    return new THREE.Color(r, g, b);
}

function shapeMesh(shape, color, values) {
    const rank = shape.length;
    const mat = new THREE.MeshStandardMaterial({ color, roughness: 0.2, metalness: 0.15 });

    if (rank === 0) {
        if (values && values.length === 1) {
            mat.color = values[0] >= 0 ? new THREE.Color(0.95, 0.5, 0.35) : new THREE.Color(0.35, 0.5, 0.95);
        }
        return new THREE.Mesh(new THREE.SphereGeometry(0.4, 20, 20), mat);
    }

    // Vector: bars recede into Z
    if (rank === 1 && values && values.length <= 64) {
        const group = new THREE.Group();
        const n = values.length;
        const min = Math.min(...values);
        const max = Math.max(...values);
        const totalD = logDim(n);
        const barD = totalD / n * 0.85;
        for (let i = 0; i < n; i++) {
            const h = Math.max(0.08, Math.abs(values[i]) / (Math.max(Math.abs(min), Math.abs(max)) || 1) * 1.0);
            const bmat = new THREE.MeshStandardMaterial({ color: valueColor(values[i], min, max), roughness: 0.2, metalness: 0.15 });
            const bar = new THREE.Mesh(new THREE.BoxGeometry(barD * 0.7, h, barD * 0.7), bmat);
            bar.position.z = -(i - n / 2) * (totalD / n);
            bar.position.y = h / 2;
            bar.castShadow = true;
            group.add(bar);
        }
        return group;
    }

    if (rank === 1) {
        const d = logDim(shape[0]);
        return new THREE.Mesh(new THREE.BoxGeometry(0.4, 0.3, d), mat);
    }

    // Matrix with values: cols across X, rows recede into Z
    if (rank === 2 && values && values.length <= 400) {
        const rows = shape[0], cols = shape[1];
        const group = new THREE.Group();
        const min = Math.min(...values);
        const max = Math.max(...values);
        const cellW = Math.max(0.08, logDim(cols) / cols);
        const cellD = Math.max(0.08, logDim(rows) / rows);
        for (let r = 0; r < rows; r++) {
            for (let c = 0; c < cols; c++) {
                const v = values[r * cols + c];
                const cmat = new THREE.MeshStandardMaterial({ color: valueColor(v, min, max), roughness: 0.2, metalness: 0.15 });
                const cell = new THREE.Mesh(new THREE.BoxGeometry(cellW * 0.88, 0.12, cellD * 0.88), cmat);
                cell.position.x = (c - cols / 2) * cellW;
                cell.position.z = -(r - rows / 2) * cellD;
                cell.position.y = 0.06;
                cell.castShadow = true;
                group.add(cell);
            }
        }
        return group;
    }

    // Matrix without values: width X, depth Z
    if (rank === 2) {
        const w = logDim(shape[1]);
        const d = logDim(shape[0]);
        return new THREE.Mesh(new THREE.BoxGeometry(w, 0.15, d), mat);
    }

    // Rank-4 conv: stacked channel heatmaps receding into Z
    if (rank === 4 && values && values.length <= 512) {
        return convChannelStack(shape, values);
    }

    // Generic rank-3+: slabs stacked into Z
    const w = logDim(shape[rank - 1] || 1);
    const h = 0.15;
    const group = new THREE.Group();
    const layers = Math.min(Math.ceil(logDim(shape[0] || 1) * 2), 8);
    const d = logDim(shape[rank - 2] || 1);
    for (let i = 0; i < layers; i++) {
        const slab = new THREE.Mesh(new THREE.BoxGeometry(w, h, d * 0.9), mat.clone());
        slab.position.z = -i * (d + 0.1);
        slab.position.y = 0.08 + i * 0.05;
        slab.castShadow = true;
        group.add(slab);
    }
    return group;
}

function convChannelStack(shape, values) {
    const [_b, c, h, w] = shape;
    const group = new THREE.Group();
    const min = Math.min(...values);
    const max = Math.max(...values);
    const gridW = logDim(w);
    const gridH = logDim(h);
    const cellW = gridW / w;
    const cellD = gridH / h;
    const spacing = 0.4;
    const maxChannels = Math.min(c, 8);
    for (let ci = 0; ci < maxChannels; ci++) {
        const channelGroup = new THREE.Group();
        const offset = ci * h * w;
        for (let r = 0; r < h; r++) {
            for (let col = 0; col < w; col++) {
                const v = values[offset + r * w + col] || 0;
                const cmat = new THREE.MeshStandardMaterial({ color: valueColor(v, min, max), roughness: 0.2, metalness: 0.15 });
                const cell = new THREE.Mesh(new THREE.BoxGeometry(cellW * 0.85, 0.1, cellD * 0.85), cmat);
                cell.position.x = (col - w / 2) * cellW;
                cell.position.z = -(r - h / 2) * cellD;
                cell.position.y = 0.05;
                channelGroup.add(cell);
            }
        }
        channelGroup.position.y = ci * spacing;
        group.add(channelGroup);
    }
    return group;
}

function drawConnections(label, targetX) {
    const rhs = label.includes('=') ? label.split('=').slice(1).join('=') : label;
    for (const [name, srcX] of Object.entries(varPositions)) {
        if (srcX === targetX) continue;
        const re = new RegExp('\\b' + name.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + '\\b');
        if (!re.test(rhs)) continue;
        const mid = (srcX + targetX) / 2;
        const curve = new THREE.QuadraticBezierCurve3(
            new THREE.Vector3(srcX, 0.3, 0),
            new THREE.Vector3(mid, 1.5, -0.5),
            new THREE.Vector3(targetX, 0.3, 0)
        );
        const geo = new THREE.TubeGeometry(curve, 20, 0.04, 6, false);
        const mat = new THREE.MeshBasicMaterial({ color: 0xffaa44, transparent: true, opacity: 0.7 });
        const tube = new THREE.Mesh(geo, mat);
        scene.add(tube);
        stepObjects.push(tube);
    }
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
    const values = ev.output?.values || null;
    const mesh = shapeMesh(shape, color, values);
    mesh.position.set(x, 0.6, 0);
    if (mesh.castShadow !== undefined) mesh.castShadow = true;
    const rawName = ev.output?.name || '';
    const isVar = ev.label.includes('=');
    const varName = isVar ? rawName : null;
    mesh.userData = { varName, label: ev.label, stepIdx: stepCount - 1, shape, elements: ev.output?.elements || 0, values, summary: ev.output?.summary || null };
    mesh.traverse(c => { if (c !== mesh) c.userData = mesh.userData; });
    scene.add(mesh);
    stepObjects.push(mesh);
    stepMeshes.push(mesh);
    prevMesh = mesh;

    if (varName) varPositions[varName] = x;
    drawConnections(ev.label, x);

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
    selectedStepIdx = -1;
    const el = document.getElementById('stage3d-detail');
    if (el) el.style.display = 'none';
}

function panToStep(idx) {
    if (!camera || idx < -3 || idx >= stepCount) return;
    const dx = (idx - viewStep) * SPACING;
    viewStep = idx;
    camera.position.x += dx;
    controls.target.x += dx;
    controls.update();
}

window.__stage3d_reset_view = function() {
    if (!camera) return;
    clearSelection();
    viewStep = 0;
    camera.position.set(0, 6, 12);
    controls.target.set(0, 0.5, 0);
    controls.update();
};

window.__stage3d_prev = function() {
    if (selectedStepIdx >= 0) { selectStep(selectedStepIdx - 1); } else { panToStep(viewStep - 1); }
};
window.__stage3d_next = function() {
    if (selectedStepIdx >= 0) { selectStep(selectedStepIdx + 1); } else { panToStep(viewStep + 1); }
};
window.__stage3d_home = function() { clearSelection(); panToStep(0); };
window.__stage3d_end = function() { clearSelection(); panToStep(stepCount - 1); };

window.__stage3d_clear = function() {
    for (const obj of stepObjects) scene.remove(obj);
    stepObjects.length = 0;
    stepMeshes.length = 0;
    selectedStepIdx = -1;
    for (const k of Object.keys(varPositions)) delete varPositions[k];
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

// Resize handle moved to inline <script> in index.html
// (ES module + CDN dep made it unreliable).
