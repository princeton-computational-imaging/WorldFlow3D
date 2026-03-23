import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { PLYLoader } from "three/addons/loaders/PLYLoader.js";

function loadMesh(src, callback, onProgress, colorOverride) {
    const ext = src.split(".").pop().toLowerCase();
    function makeMaterial(geom) {
        if (colorOverride) {
            return new THREE.MeshStandardMaterial({
                color: new THREE.Color(colorOverride),
                roughness: 1.0, metalness: 0.0, side: THREE.DoubleSide,
            });
        }
        const colorAttr = geom.hasAttribute("color") ? "color"
            : geom.hasAttribute("_color") ? "_color" : null;
        if (colorAttr) {
            if (colorAttr === "_color") geom.setAttribute("color", geom.getAttribute("_color"));
            return new THREE.MeshStandardMaterial({
                vertexColors: true, roughness: 1.0, metalness: 0.0, side: THREE.DoubleSide,
            });
        }
        return new THREE.MeshStandardMaterial({
            color: 0x8899aa, roughness: 1.0, metalness: 0.0, side: THREE.DoubleSide,
        });
    }
    if (ext === "ply") {
        new PLYLoader().load(src, (geometry) => {
            geometry.computeVertexNormals();
            callback(new THREE.Mesh(geometry, makeMaterial(geometry)));
        }, onProgress, (err) => console.error("[mesh-viewer]", err));
    } else {
        new GLTFLoader().load(src, (gltf) => {
            gltf.scene.traverse((child) => {
                if (!child.isMesh) return;
                child.geometry.computeVertexNormals();
                child.material = makeMaterial(child.geometry);
            });
            callback(gltf.scene);
        }, onProgress, (err) => console.error("[mesh-viewer]", err));
    }
}

function normalizeMesh(object) {
    // Tilt 90° around X to convert Z-up to Y-up
    const tilt = new THREE.Group();
    tilt.add(object);
    tilt.rotation.x = -Math.PI / 2;

    // Compute bounds after tilt
    tilt.updateMatrixWorld(true);
    const box = new THREE.Box3().setFromObject(tilt);
    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3());
    const maxDim = Math.max(size.x, size.y, size.z);

    const pivot = new THREE.Group();
    pivot.add(tilt);
    tilt.position.set(-center.x, -center.y, -center.z);
    const s = 2.0 / maxDim;
    pivot.scale.set(s, s, s);
    return pivot;
}

function createScene() {
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xcccccc);

    scene.add(new THREE.AmbientLight(0xffffff, 0.2));

    // Main sun: 4644K warm white (#FFD8C1)
    const sun = new THREE.DirectionalLight(0xffd8c1, 1.5);
    sun.position.set(5, 10, 7);
    scene.add(sun);

    // Hard shadow sun: cooler, from a different angle
    const hardSun = new THREE.DirectionalLight(0xffffff, 0.8);
    hardSun.position.set(-3, 8, 4);
    scene.add(hardSun);
    return scene;
}

function initMeshViewer(container) {
    // Parse scene configs
    let scenes;
    if (container.dataset.scenes) {
        scenes = JSON.parse(container.dataset.scenes);
    } else if (container.dataset.src) {
        scenes = [{ src: container.dataset.src, srcAlt: container.dataset.srcAlt, label: container.dataset.label || "A", labelAlt: container.dataset.labelAlt || "B" }];
    } else {
        return;
    }

    const width = container.clientWidth;
    const height = container.clientHeight;
    if (width === 0 || height === 0) return;

    const camera = new THREE.PerspectiveCamera(45, width / height, 0.001, 100);
    camera.position.set(1.25, 0.75, 1.25);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.2;
    container.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.autoRotate = true;
    controls.autoRotateSpeed = 1.5;
    controls.target.set(0, 0, 0);

    // Loading overlay
    const loadingEl = document.createElement("div");
    loadingEl.className = "mesh-loading-overlay";
    loadingEl.innerHTML = '<div class="large-loading-spinner"></div><div class="loading-progress">Loading mesh...</div>';
    container.appendChild(loadingEl);

    // Compare UI elements
    let compareMode = false;
    let splitX = 0.5;

    const dividerEl = document.createElement("div");
    dividerEl.className = "mesh-divider";
    dividerEl.style.display = "none";
    container.appendChild(dividerEl);

    const toggleBtn = document.createElement("button");
    toggleBtn.className = "mesh-compare-toggle";
    toggleBtn.style.display = "none";
    container.appendChild(toggleBtn);

    const labelLeft = document.createElement("div");
    labelLeft.className = "mesh-split-label mesh-split-label-left";
    labelLeft.style.display = "none";
    container.appendChild(labelLeft);

    const labelRight = document.createElement("div");
    labelRight.className = "mesh-split-label mesh-split-label-right";
    labelRight.style.display = "none";
    container.appendChild(labelRight);

    // Divider drag
    let dragging = false;
    dividerEl.addEventListener("mousedown", (e) => { dragging = true; e.preventDefault(); });
    dividerEl.addEventListener("touchstart", () => { dragging = true; }, { passive: true });
    const onMove = (clientX) => {
        if (!dragging) return;
        const rect = container.getBoundingClientRect();
        splitX = Math.max(0.05, Math.min(0.95, (clientX - rect.left) / rect.width));
        dividerEl.style.left = `${splitX * 100}%`;
        labelLeft.style.left = `${splitX * 50}%`;
        labelRight.style.left = `${splitX * 100 + (100 - splitX * 100) / 2}%`;
    };
    window.addEventListener("mousemove", (e) => onMove(e.clientX));
    window.addEventListener("touchmove", (e) => onMove(e.touches[0].clientX));
    window.addEventListener("mouseup", () => { dragging = false; });
    window.addEventListener("touchend", () => { dragging = false; });

    // Scene cache: { sceneA, sceneB, loaded }
    const sceneCache = scenes.map(() => ({ sceneA: null, sceneB: null, loaded: false }));
    let activeIndex = 0;

    function showLoading(text) {
        loadingEl.style.display = "flex";
        loadingEl.querySelector(".loading-progress").textContent = text;
    }

    function hideLoading() {
        loadingEl.style.display = "none";
    }

    function loadSceneAtIndex(idx) {
        const config = scenes[idx];
        const entry = sceneCache[idx];
        if (entry.loaded) {
            hideLoading();
            return;
        }

        showLoading(`Loading ${config.label || "scene"}...`);

        entry.sceneA = createScene();
        let aLoaded = false;
        let bLoaded = !config.srcAlt;

        function checkDone() {
            if (aLoaded && bLoaded) {
                entry.loaded = true;
                if (activeIndex === idx) hideLoading();
            }
        }

        loadMesh(config.src, (object) => {
            entry.sceneA.add(normalizeMesh(object));
            aLoaded = true;
            checkDone();
        }, (xhr) => {
            if (xhr.total && activeIndex === idx) {
                showLoading(`Loading ${config.label || "scene"}... ${Math.round(xhr.loaded / xhr.total * 100)}%`);
            }
        }, config.color || null);

        if (config.srcAlt) {
            entry.sceneB = createScene();
            loadMesh(config.srcAlt, (object) => {
                entry.sceneB.add(normalizeMesh(object));
                bLoaded = true;
                checkDone();
            }, (xhr) => {
                if (xhr.total && aLoaded && activeIndex === idx) {
                    showLoading(`Loading ${config.labelAlt || "alt"}... ${Math.round(xhr.loaded / xhr.total * 100)}%`);
                }
            }, config.colorAlt || config.color || null);
        }
    }

    function switchScene(idx) {
        activeIndex = idx;
        const config = scenes[idx];

        // Default to compare mode if alt source exists
        if (config.srcAlt) {
            compareMode = true;
            renderer.setScissorTest(true);
            dividerEl.style.display = "block";
            labelLeft.style.display = "block";
            labelRight.style.display = "block";
            splitX = 0.5;
            dividerEl.style.left = "50%";
            labelLeft.style.left = "25%";
            labelRight.style.left = "75%";
            toggleBtn.style.display = "block";
            toggleBtn.textContent = "Hide comparison";
            labelLeft.textContent = config.label || "A";
            labelRight.textContent = config.labelAlt || "B";
        } else {
            compareMode = false;
            renderer.setScissorTest(false);
            dividerEl.style.display = "none";
            labelLeft.style.display = "none";
            labelRight.style.display = "none";
            toggleBtn.style.display = "none";
        }

        // Reset camera
        camera.position.set(1.25, 0.75, 1.25);
        controls.target.set(0, 0, 0);
        camera.lookAt(0, 0, 0);
        controls.update();

        loadSceneAtIndex(idx);
    }

    // Toggle compare
    toggleBtn.addEventListener("click", () => {
        compareMode = !compareMode;
        renderer.setScissorTest(compareMode);
        dividerEl.style.display = compareMode ? "block" : "none";
        labelLeft.style.display = compareMode ? "block" : "none";
        labelRight.style.display = compareMode ? "block" : "none";
        const config = scenes[activeIndex];
        toggleBtn.textContent = compareMode ? "Hide comparison" : `Compare: ${config.labelAlt}`;
        // Reset divider position
        splitX = 0.5;
        dividerEl.style.left = "50%";
        labelLeft.style.left = "25%";
        labelRight.style.left = "75%";
    });

    // Wire up scene tabs
    const tabContainer = container.closest(".column")?.querySelector(".mesh-scene-tabs");
    if (tabContainer) {
        tabContainer.querySelectorAll(".mesh-scene-tab").forEach((tab) => {
            tab.addEventListener("click", () => {
                tabContainer.querySelectorAll(".mesh-scene-tab").forEach(t => t.classList.remove("active"));
                tab.classList.add("active");
                switchScene(parseInt(tab.dataset.scene, 10));
            });
        });
    }

    // Render loop
    function animate() {
        requestAnimationFrame(animate);
        controls.update();

        const entry = sceneCache[activeIndex];
        if (!entry.sceneA) return;

        const w = container.clientWidth;
        const h = container.clientHeight;

        if (compareMode && entry.sceneB) {
            renderer.setScissorTest(true);
            const splitPx = Math.round(splitX * w);

            renderer.setViewport(0, 0, w, h);
            renderer.setScissor(0, 0, splitPx, h);
            renderer.render(entry.sceneA, camera);

            renderer.setScissor(splitPx, 0, w - splitPx, h);
            renderer.render(entry.sceneB, camera);
        } else {
            renderer.setScissorTest(false);
            renderer.render(entry.sceneA, camera);
        }
    }
    animate();

    // Resize
    const resizeObserver = new ResizeObserver(() => {
        const w = container.clientWidth;
        const h = container.clientHeight;
        if (w === 0 || h === 0) return;
        camera.aspect = w / h;
        camera.updateProjectionMatrix();
        renderer.setSize(w, h);
    });
    resizeObserver.observe(container);

    // Auto-rotate pause
    let idleTimeout;
    controls.addEventListener("start", () => { controls.autoRotate = false; clearTimeout(idleTimeout); });
    controls.addEventListener("end", () => { idleTimeout = setTimeout(() => { controls.autoRotate = true; }, 3000); });

    // Load first scene
    switchScene(0);
}

document.querySelectorAll(".mesh-viewer").forEach(initMeshViewer);
