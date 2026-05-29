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
    new THREE.TextureLoader().load('salt-flat-pano.jpg?ts=1779917889011', tex => {
        const aspect = tex.image.width / tex.image.height;
        const h = 200;
        const w = h * aspect * 6;
        const mat = new THREE.MeshBasicMaterial({ map: tex, fog: false });
        const plane = new THREE.Mesh(new THREE.PlaneGeometry(w, h), mat);
        plane.position.set(0, 18, -160);
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
    // Far mountains: shorter, wider, packed shoulder-to-shoulder
    // so no sky shows between them.
    const farMat = new THREE.MeshStandardMaterial({ color: 0x8B6B4A, flatShading: true });
    const nearMat = new THREE.MeshStandardMaterial({ color: 0x5A8A4A, flatShading: true });
    for (let i = -100; i < 100; i++) {
        const x = i * 8 + (Math.random() - 0.5) * 2;
        const h = 5 + Math.random() * 5;
        const w = 12 + Math.random() * 6;
        const far = new THREE.Mesh(new THREE.ConeGeometry(w, h, 5 + Math.floor(Math.random() * 3)), farMat);
        far.position.set(x, h / 2, -60 - Math.random() * 10);
        scene.add(far);
    }
    // Near hills: shorter still, wider, no gaps, pushed back to
    // sit right in front of the mountains (was z=-45..-57, now
    // z=-55..-58 so the green band hugs the brown band).
    for (let i = -100; i < 100; i++) {
        const x = i * 6 + (Math.random() - 0.5) * 1.5;
        const h = 2 + Math.random() * 2.5;
        const w = 9 + Math.random() * 4;
        const near = new THREE.Mesh(new THREE.ConeGeometry(w, h, 4 + Math.floor(Math.random() * 3)), nearMat);
        near.position.set(x, h / 2, -55 - Math.random() * 3);
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
    // Pulse the selection pointer to advertise that it's
    // clickable. Sine wave on emissiveIntensity over ~1.4s;
    // also gently bobs the cone up and down so the affordance
    // catches the eye in peripheral vision.
    if (selectionPointer) {
        const phase = (now - selectionPointerStart) / 1400;
        const s = (Math.sin(phase * Math.PI * 2) + 1) * 0.5; // 0..1
        if (selectionPointer.material) {
            selectionPointer.material.emissiveIntensity = 0.45 + s * 0.55;
        }
        selectionPointer.position.y = selectionPointerBaseY + s * 0.12;
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
    // Inspector dialog: close button + backdrop close. Both
    // re-bound on every init in case the DOM nodes were
    // re-rendered.
    const closeBtn = document.getElementById('stage3d-inspector-close');
    if (closeBtn) closeBtn.addEventListener('click', closeInspector);
    const backdrop = document.getElementById('stage3d-inspector-backdrop');
    if (backdrop) backdrop.addEventListener('click', closeInspector);
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
    // Hit-test the selection pointer first so it short-circuits
    // any sculpture under or behind it: a click on the yellow
    // marker opens the close-up inspector for the currently
    // selected mesh.
    if (selectionPointer) {
        const pHits = raycaster.intersectObject(selectionPointer, false);
        if (pHits.length > 0) {
            openInspector();
            return;
        }
    }
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
// Anchored base Y + start timestamp drive the bob + glow
// animation in animate(); reset whenever a new mesh becomes
// the selection.
let selectionPointerBaseY = 0;
let selectionPointerStart = 0;

function selectMesh(mesh) {
    if (selectionPointer) { scene.remove(selectionPointer); selectionPointer = null; }
    const worldPos = new THREE.Vector3();
    const target = mesh.parent?.isGroup ? mesh.parent : mesh;
    target.getWorldPosition(worldPos);
    // Pointer sized up a touch so it's a comfortable click target.
    const geo = new THREE.ConeGeometry(0.22, 0.55, 4);
    geo.rotateX(Math.PI);
    const mat = new THREE.MeshStandardMaterial({ color: 0xffcc44, emissive: 0xffcc44, emissiveIntensity: 0.6 });
    selectionPointer = new THREE.Mesh(geo, mat);
    selectionPointer.position.set(worldPos.x, worldPos.y + 1.5, worldPos.z);
    selectionPointer.userData = { isSelectionPointer: true, sourceMesh: mesh };
    scene.add(selectionPointer);
    // Reset the pulse animation so each fresh selection starts
    // at the brightest point of the cycle -- bigger visual
    // pop the moment the user clicks something.
    selectionPointerBaseY = worldPos.y + 1.5;
    selectionPointerStart = performance.now();
    selectedStepIdx = mesh.userData?.stepIdx ?? -1;
}

// Close-up inspector dialog. Opens with the currently selected
// sculpture's tensor info in a centered modal panel. The full
// 3D close-up scene + interactive axis-slicing tour are tracked
// in docs/3d-introspect-dialog.md; this is the v0 starter that
// makes the yellow pointer clickable and surfaces the existing
// detail data + a sampled values dump in one clean dialog.
// Viz-IR renderer registry. Saga A (viz-ir-scaffold) ships the
// dispatch site; sagas B+ register one entry per VizKind here.
// Empty kinds fall through to the existing text body.
const vizRenderers = Object.create(null);

// Saga B: detect tensors that are attention weights and stamp
// a VizNode-shaped object onto userData. JS-side heuristic for
// now; richer evaluator-driven detection lands in saga C when
// real tokens come into scope. The shape of the object mirrors
// mlpl-web-viz-ir's VizNode + AttentionViz so future Rust-side
// renderers can swap in without touching the JS detection.
const SOFTMAX_RE = /^\s*[A-Za-z_][A-Za-z0-9_]*\s*=\s*softmax\s*\(/;
function detectViz(label, shape, values) {
    // Saga B debug -- remove once verified. Log every event so
    // the user can see in DevTools whether the detection ever
    // gets the inputs it needs. Turn off via window.__viz_debug = false.
    if (window.__viz_debug !== false) {
        console.log('[viz-ir] detectViz', {
            label,
            shape,
            values_len: values ? values.length : null,
            matched: !!(values && shape.length === 2 && shape[0] === shape[1]
                && SOFTMAX_RE.test(label || '')),
        });
    }
    if (!label || !shape || !values) return null;
    // Rank-2 square + a softmax assignment is the attention.mlpl
    // pattern (and the prevailing convention even outside it).
    if (shape.length === 2 && shape[0] === shape[1] && shape[0] >= 2
        && values.length === shape[0] * shape[1]
        && SOFTMAX_RE.test(label)) {
        const n = shape[0];
        const tokens = Array.from({ length: n }, (_, i) => ({ index: i }));
        return {
            kind: 'attention',
            attention: {
                query_tokens: tokens,
                key_tokens: tokens,
                weights: Array.from(values),
                layout: { rank: 'qk', q: n, k: n },
                causal: false,
            },
        };
    }
    return null;
}

// Viridis ramp (matches mlpl-viz/src/svg/heatmap_grid.rs).
function viridis(t) {
    const stops = [[68, 1, 84], [33, 145, 140], [253, 231, 37]];
    const [a, b, f] = t < 0.5 ? [stops[0], stops[1], t * 2] : [stops[1], stops[2], (t - 0.5) * 2];
    const r = Math.round(a[0] + (b[0] - a[0]) * f);
    const g = Math.round(a[1] + (b[1] - a[1]) * f);
    const bl = Math.round(a[2] + (b[2] - a[2]) * f);
    return `rgb(${r}, ${g}, ${bl})`;
}

function tokenLabel(t) {
    if (t == null) return '';
    if (typeof t === 'string') return t;
    if (typeof t.str === 'string') return t.str;
    if (typeof t.index === 'number') return String(t.index);
    return '';
}

function renderAttentionHeatmap(ud, viz) {
    const a = viz.attention;
    if (!a || !a.weights || !a.weights.length) {
        return '<div class="insp-hint">attention payload missing</div>';
    }
    const qLabels = (a.query_tokens || []).map(tokenLabel);
    const kLabels = (a.key_tokens || []).map(tokenLabel);
    const q = qLabels.length;
    const k = kLabels.length;
    // Sample one matrix when the payload has head/batch dimensions.
    const head = a.head || 0;
    const layerHeadOffset = head * q * k;
    const cellW = Math.max(20, Math.min(48, Math.floor(560 / Math.max(k, 8))));
    const cellH = cellW;
    const padL = 36, padT = 24, padR = 90;
    const w = padL + k * cellW + padR;
    const h = padT + q * cellH + 30;
    let svg = `<svg viewBox="0 0 ${w} ${h}" width="100%" style="max-width:680px;background:var(--mantle);border-radius:6px">`;
    // Key axis labels (top).
    for (let j = 0; j < k; j++) {
        const x = padL + j * cellW + cellW / 2;
        svg += `<text x="${x}" y="${padT - 8}" text-anchor="middle" font-size="11" font-family="var(--mono)" fill="var(--subtext1)">${escapeHtml(kLabels[j])}</text>`;
    }
    // Cells + row labels (left).
    let maxRowSum = 0;
    const rowSums = new Float64Array(q);
    for (let i = 0; i < q; i++) {
        let rowSum = 0;
        for (let j = 0; j < k; j++) {
            const v = a.weights[layerHeadOffset + i * k + j] || 0;
            rowSum += v;
            const x = padL + j * cellW;
            const y = padT + i * cellH;
            const col = viridis(Math.max(0, Math.min(1, v)));
            const tip = `(${escapeHtml(qLabels[i])}, ${escapeHtml(kLabels[j])}) = ${v.toFixed(4)}`;
            svg += `<rect x="${x}" y="${y}" width="${cellW}" height="${cellH}" fill="${col}" stroke="rgba(0,0,0,0.15)" stroke-width="0.5"><title>${tip}</title></rect>`;
        }
        rowSums[i] = rowSum;
        if (rowSum > maxRowSum) maxRowSum = rowSum;
        const ry = padT + i * cellH + cellH / 2 + 4;
        svg += `<text x="${padL - 6}" y="${ry}" text-anchor="end" font-size="11" font-family="var(--mono)" fill="var(--subtext1)">${escapeHtml(qLabels[i])}</text>`;
    }
    // Row-sum bars (right) so the user can see softmax normalization.
    const barBase = padL + k * cellW + 8;
    const barW = padR - 16;
    for (let i = 0; i < q; i++) {
        const y = padT + i * cellH + cellH * 0.2;
        const bh = cellH * 0.6;
        const frac = maxRowSum > 0 ? rowSums[i] / Math.max(1, maxRowSum) : 0;
        svg += `<rect x="${barBase}" y="${y}" width="${barW * frac}" height="${bh}" fill="var(--peach)" opacity="0.6"><title>row sum = ${rowSums[i].toFixed(4)}</title></rect>`;
        svg += `<text x="${barBase + barW + 4}" y="${y + bh / 2 + 4}" text-anchor="start" font-size="10" font-family="var(--mono)" fill="var(--subtext0)">${rowSums[i].toFixed(2)}</text>`;
    }
    // Axis titles.
    svg += `<text x="${padL + k * cellW / 2}" y="${h - 8}" text-anchor="middle" font-size="11" font-family="var(--mono)" fill="var(--subtext0)">key tokens</text>`;
    svg += `<text x="12" y="${padT + q * cellH / 2}" text-anchor="middle" font-size="11" font-family="var(--mono)" fill="var(--subtext0)" transform="rotate(-90 12 ${padT + q * cellH / 2})">query tokens</text>`;
    svg += `</svg>`;

    const name = ud.varName || '';
    const headline = `${name ? name + ' = ' : ''}${ud.label || ''}`;
    return `
        <h2>${escapeHtml(headline)}</h2>
        <div class="insp-row"><strong>Attention pattern</strong> &nbsp; <strong>Shape:</strong> [${q}, ${k}] &nbsp; (row-wise softmax)</div>
        <div class="insp-row" style="margin-top:14px">${svg}</div>
        <div class="insp-section-title">Statistics</div>
        ${ud.values && ud.values.length ? renderStats(computeStats(ud.values)) : ''}
        <div class="insp-hint">Saga B v0: integer indices as labels; saga C threads BPE tokens. Saga F adds D3 hover-to-trace.</div>
    `;
}

// Register the renderer so renderInspectorBody picks it up when
// userData.viz.kind === 'attention'. window.__viz_register is
// also exposed for renderers that live outside this file.
vizRenderers['attention'] = renderAttentionHeatmap;

function openInspector() {
    if (selectedStepIdx < 0 || selectedStepIdx >= stepMeshes.length) return;
    const mesh = stepMeshes[selectedStepIdx];
    if (!mesh || !mesh.userData) return;
    const dlg = document.getElementById('stage3d-inspector');
    const body = document.getElementById('stage3d-inspector-body');
    if (!dlg || !body) return;
    body.innerHTML = renderInspectorBody(mesh.userData);
    dlg.style.display = 'block';
    // ESC closes; one listener per open so we can detach on close.
    const onKey = (e) => { if (e.key === 'Escape') closeInspector(); };
    document.addEventListener('keydown', onKey);
    dlg.dataset.escListener = '1';
    dlg._escHandler = onKey;
}

function closeInspector() {
    const dlg = document.getElementById('stage3d-inspector');
    if (!dlg) return;
    dlg.style.display = 'none';
    if (dlg._escHandler) {
        document.removeEventListener('keydown', dlg._escHandler);
        delete dlg._escHandler;
    }
    // Restore focus to the canvas so arrow keys resume working
    // on the selected mesh without the user having to click
    // back into the 3D space.
    //
    // Deferred to the next animation frame because Chromium
    // restores focus to the document body AFTER the current
    // click event finishes propagating; calling focus() here
    // synchronously gets immediately overridden. requestAnimationFrame
    // sidesteps that by waiting for the focus restoration to
    // settle. activeElement.blur() drops the close-button or
    // backdrop from the focus ring first so the rAF focus()
    // isn't competing with a delayed steal.
    if (document.activeElement && document.activeElement !== document.body) {
        try { document.activeElement.blur(); } catch (_) {}
    }
    const canvas = renderer && renderer.domElement;
    if (canvas && typeof canvas.focus === 'function') {
        // Belt-and-suspenders: confirm tabIndex is still
        // present (would be lost if Yew replaced the canvas
        // node) before asking the browser to focus it.
        if (!canvas.hasAttribute('tabindex')) canvas.tabIndex = 0;
        requestAnimationFrame(() => {
            try { canvas.focus({ preventScroll: true }); } catch (_) {}
        });
    }
}

function renderInspectorBody(ud) {
    // Viz-IR dispatch: if a renderer is registered for this
    // sculpture's viz.kind, use it; otherwise fall through to
    // the existing text body. Saga A registers nothing -- the
    // type field rides through but doesn't change UX.
    const viz = ud.viz;
    if (viz && typeof viz.kind === 'string' && vizRenderers[viz.kind]) {
        try {
            return vizRenderers[viz.kind](ud, viz);
        } catch (e) {
            console.warn('viz-ir renderer failed; falling back to text body', e);
        }
    }
    const name = ud.varName || '';
    const label = ud.label || '';
    const shape = ud.shape || [];
    const dims = shape.length ? '[' + shape.join(', ') + ']' : 'scalar';
    const rank = shape.length;
    const elements = ud.elements || (shape.length ? shape.reduce((a, b) => a * b, 1) : 1);
    const bytes = elements * 8;
    const mem = bytes < 1024 ? `${bytes} B`
              : bytes < 1048576 ? `${(bytes / 1024).toFixed(1)} KB`
              : `${(bytes / 1048576).toFixed(1)} MB`;
    const headline = `${name ? name + ' = ' : ''}${label}`;
    let html = `<h2>${escapeHtml(headline)}</h2>`;
    html += `<div class="insp-row"><strong>Shape:</strong> ${dims} &nbsp; <strong>Rank:</strong> ${rank} &nbsp; <strong>Elements:</strong> ${elements.toLocaleString()} &nbsp; <strong>Memory:</strong> ~${mem}</div>`;
    html += `<div class="insp-row"><strong>Step:</strong> ${ud.stepIdx !== undefined ? ud.stepIdx + 1 : '?'} of ${stepCount}</div>`;
    if (ud.values && ud.values.length) {
        const stats = computeStats(ud.values);
        html += `<div class="insp-section-title">Statistics</div>` + renderStats(stats);
        html += `<div class="insp-section-title">Values (first ${Math.min(ud.values.length, 64)})</div>`;
        html += `<div class="insp-values">${formatInspectorValues(ud.values, 64)}</div>`;
    } else if (ud.summary) {
        html += `<div class="insp-section-title">Statistics</div>` + renderStats(ud.summary);
        html += `<div class="insp-hint">Tensor is too large for an inline values dump; statistics shown above are computed from the full set.</div>`;
    }
    html += `<div class="insp-hint">Interactive 3D close-up + axis-label slicing + drill-down for composite objects is queued (see docs/3d-introspect-dialog.md).</div>`;
    return html;
}

function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, c => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
    }[c]));
}

function formatInspectorValues(vals, limit) {
    const n = Math.min(vals.length, limit);
    const out = [];
    for (let i = 0; i < n; i++) {
        out.push(vals[i].toFixed(4).padStart(10));
        if ((i + 1) % 8 === 0) out.push('\n');
        else out.push(' ');
    }
    if (vals.length > limit) out.push(`\n... (${(vals.length - limit).toLocaleString()} more)`);
    return out.join('').trimEnd();
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
    const viz = detectViz(ev.label, shape, values);
    mesh.userData = { varName, label: ev.label, stepIdx: stepCount - 1, shape, elements: ev.output?.elements || 0, values, summary: ev.output?.summary || null, viz };
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
window.__stage3d_close_inspector = closeInspector;
// Saga A (viz-ir-scaffold): future renderer sagas register one
// function per VizKind without having to edit this file.
//   window.__viz_register('attention', (userData, vizNode) => '<svg>...</svg>');
// The dispatch site in renderInspectorBody picks the matching
// renderer when present; absent kinds fall through to the
// existing text body.
window.__viz_register = function(kind, renderer) {
    if (typeof kind === 'string' && typeof renderer === 'function') {
        vizRenderers[kind] = renderer;
    }
};

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
