import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { PLYLoader } from "three/addons/loaders/PLYLoader.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";

/**
 * Simple interactive mesh viewer for PLY/GLB files.
 * Usage: add a <div class="mesh-viewer" data-src="path/to/mesh.ply"></div>
 */

function initMeshViewer(container) {
    const src = container.dataset.src;
    if (!src) return;

    const width = container.clientWidth;
    const height = container.clientHeight;

    // Scene
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x1a1a2e);

    // Camera
    const camera = new THREE.PerspectiveCamera(45, width / height, 0.01, 1000);
    camera.position.set(0, 1, 3);

    // Renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.0;
    container.appendChild(renderer.domElement);

    // Controls
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.autoRotate = true;
    controls.autoRotateSpeed = 1.5;

    // Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    scene.add(ambientLight);

    const dirLight = new THREE.DirectionalLight(0xffffff, 1.0);
    dirLight.position.set(5, 10, 7);
    scene.add(dirLight);

    const dirLight2 = new THREE.DirectionalLight(0xffffff, 0.4);
    dirLight2.position.set(-5, 5, -5);
    scene.add(dirLight2);

    // Loading indicator
    const loadingEl = document.createElement("div");
    loadingEl.className = "mesh-loading-overlay";
    loadingEl.innerHTML = '<div class="large-loading-spinner"></div><div class="loading-progress">Loading mesh...</div>';
    container.appendChild(loadingEl);

    // Load mesh
    const ext = src.split(".").pop().toLowerCase();

    function onMeshLoaded(object) {
        // Center and scale the mesh to fit the view
        const box = new THREE.Box3().setFromObject(object);
        const center = box.getCenter(new THREE.Vector3());
        const size = box.getSize(new THREE.Vector3());
        const maxDim = Math.max(size.x, size.y, size.z);
        const scale = 2.0 / maxDim;

        object.position.sub(center);
        object.scale.multiplyScalar(scale);
        scene.add(object);

        controls.target.set(0, 0, 0);
        camera.position.set(0, 1, 3);
        controls.update();

        // Remove loading indicator
        loadingEl.remove();
    }

    if (ext === "ply") {
        const loader = new PLYLoader();
        loader.load(
            src,
            (geometry) => {
                geometry.computeVertexNormals();
                const hasColors = geometry.hasAttribute("color");
                const material = new THREE.MeshStandardMaterial({
                    vertexColors: hasColors,
                    color: hasColors ? 0xffffff : 0x8899aa,
                    roughness: 0.6,
                    metalness: 0.1,
                    side: THREE.DoubleSide,
                });
                const mesh = new THREE.Mesh(geometry, material);
                onMeshLoaded(mesh);
            },
            (xhr) => {
                if (xhr.total > 0) {
                    const pct = Math.round((xhr.loaded / xhr.total) * 100);
                    loadingEl.querySelector(".loading-progress").textContent = `Loading mesh... ${pct}%`;
                }
            },
            (err) => {
                console.error("Error loading PLY:", err);
                loadingEl.querySelector(".loading-progress").textContent = "Failed to load mesh";
            }
        );
    } else if (ext === "glb" || ext === "gltf") {
        const loader = new GLTFLoader();
        loader.load(
            src,
            (gltf) => onMeshLoaded(gltf.scene),
            (xhr) => {
                if (xhr.total > 0) {
                    const pct = Math.round((xhr.loaded / xhr.total) * 100);
                    loadingEl.querySelector(".loading-progress").textContent = `Loading mesh... ${pct}%`;
                }
            },
            (err) => {
                console.error("Error loading GLB:", err);
                loadingEl.querySelector(".loading-progress").textContent = "Failed to load mesh";
            }
        );
    }

    // Animation loop
    function animate() {
        requestAnimationFrame(animate);
        controls.update();
        renderer.render(scene, camera);
    }
    animate();

    // Resize handling
    const resizeObserver = new ResizeObserver(() => {
        const w = container.clientWidth;
        const h = container.clientHeight;
        camera.aspect = w / h;
        camera.updateProjectionMatrix();
        renderer.setSize(w, h);
    });
    resizeObserver.observe(container);

    // Pause auto-rotate on interaction, resume after idle
    let idleTimeout;
    controls.addEventListener("start", () => {
        controls.autoRotate = false;
        clearTimeout(idleTimeout);
    });
    controls.addEventListener("end", () => {
        idleTimeout = setTimeout(() => {
            controls.autoRotate = true;
        }, 3000);
    });
}

// Initialize all mesh viewers on page load
document.addEventListener("DOMContentLoaded", () => {
    document.querySelectorAll(".mesh-viewer").forEach(initMeshViewer);
});
