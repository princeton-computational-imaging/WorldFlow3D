import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { PLYLoader } from "three/addons/loaders/PLYLoader.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";

function initMeshViewer(container) {
    const src = container.dataset.src;
    if (!src) return;

    const width = container.clientWidth;
    const height = container.clientHeight;
    if (width === 0 || height === 0) return;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xcccccc);

    const camera = new THREE.PerspectiveCamera(45, width / height, 0.001, 100);

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

    scene.add(new THREE.AmbientLight(0xffffff, 0.8));
    const dirLight = new THREE.DirectionalLight(0xffffff, 1.2);
    dirLight.position.set(5, 10, 7);
    scene.add(dirLight);
    const dirLight2 = new THREE.DirectionalLight(0xffffff, 0.5);
    dirLight2.position.set(-5, 5, -5);
    scene.add(dirLight2);

    const loadingEl = document.createElement("div");
    loadingEl.className = "mesh-loading-overlay";
    loadingEl.innerHTML = '<div class="large-loading-spinner"></div><div class="loading-progress">Loading mesh...</div>';
    container.appendChild(loadingEl);

    function fitCameraToObject(object) {
        const box = new THREE.Box3().setFromObject(object);
        const center = box.getCenter(new THREE.Vector3());
        const size = box.getSize(new THREE.Vector3());
        const maxDim = Math.max(size.x, size.y, size.z);

        // Normalize: move center to origin, scale to unit size
        const pivot = new THREE.Group();
        pivot.add(object);
        object.position.set(-center.x, -center.y, -center.z);
        const s = 2.0 / maxDim;
        pivot.scale.set(s, s, s);
        scene.add(pivot);

        // Place camera
        camera.position.set(2.5, 1.5, 2.5);
        controls.target.set(0, 0, 0);
        camera.lookAt(0, 0, 0);
        controls.update();

        loadingEl.remove();
    }

    const ext = src.split(".").pop().toLowerCase();

    if (ext === "ply") {
        new PLYLoader().load(
            src,
            (geometry) => {
                geometry.computeVertexNormals();
                const mat = new THREE.MeshStandardMaterial({
                    vertexColors: geometry.hasAttribute("color"),
                    color: geometry.hasAttribute("color") ? 0xffffff : 0x8899aa,
                    roughness: 0.6, metalness: 0.1, side: THREE.DoubleSide,
                });
                fitCameraToObject(new THREE.Mesh(geometry, mat));
            },
            (xhr) => { if (xhr.total) loadingEl.querySelector(".loading-progress").textContent = `Loading... ${Math.round(xhr.loaded / xhr.total * 100)}%`; },
            (err) => { console.error("[mesh-viewer]", err); loadingEl.querySelector(".loading-progress").textContent = "Failed to load"; }
        );
    } else if (ext === "glb" || ext === "gltf") {
        new GLTFLoader().load(
            src,
            (gltf) => {
                gltf.scene.traverse((child) => {
                    if (!child.isMesh) return;
                    const geom = child.geometry;
                    // GLB vertex colors can be "color" or "_color"
                    const colorAttr = geom.hasAttribute("color") ? "color"
                        : geom.hasAttribute("_color") ? "_color" : null;
                    if (colorAttr) {
                        if (colorAttr === "_color") {
                            geom.setAttribute("color", geom.getAttribute("_color"));
                        }
                        child.material = new THREE.MeshStandardMaterial({
                            vertexColors: true,
                            roughness: 0.6, metalness: 0.1, side: THREE.DoubleSide,
                        });
                    } else {
                        child.material = new THREE.MeshStandardMaterial({
                            color: 0x8899aa,
                            roughness: 0.6, metalness: 0.1, side: THREE.DoubleSide,
                        });
                    }
                    geom.computeVertexNormals();
                });
                fitCameraToObject(gltf.scene);
            },
            (xhr) => { if (xhr.total) loadingEl.querySelector(".loading-progress").textContent = `Loading... ${Math.round(xhr.loaded / xhr.total * 100)}%`; },
            (err) => { console.error("[mesh-viewer]", err); loadingEl.querySelector(".loading-progress").textContent = "Failed to load"; }
        );
    }

    function animate() {
        requestAnimationFrame(animate);
        controls.update();
        renderer.render(scene, camera);
    }
    animate();

    const resizeObserver = new ResizeObserver(() => {
        const w = container.clientWidth;
        const h = container.clientHeight;
        if (w === 0 || h === 0) return;
        camera.aspect = w / h;
        camera.updateProjectionMatrix();
        renderer.setSize(w, h);
    });
    resizeObserver.observe(container);

    let idleTimeout;
    controls.addEventListener("start", () => { controls.autoRotate = false; clearTimeout(idleTimeout); });
    controls.addEventListener("end", () => { idleTimeout = setTimeout(() => { controls.autoRotate = true; }, 3000); });
}

document.querySelectorAll(".mesh-viewer").forEach(initMeshViewer);
