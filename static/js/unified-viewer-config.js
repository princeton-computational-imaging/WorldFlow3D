// Unified Viewer Configuration
// This file contains shared settings and initialization logic for both main carousel and individual scene pages
import * as THREE from "three";
import { SparkControls, SplatMesh, SparkRenderer, SplatLoader } from "@sparkjsdev/spark";

// Camera bounds configuration (copied from main script for consistency)
const CAMERA_BOUNDS = {
    // Position bounds
    position: {
        x: { min: -100, max: 100 },
        y: { min: -100, max: 100 },
        z: { min: 0.1, max: 300 }
    },
    // Rotation bounds (in radians)
    rotation: {
        x: { min: -Math.PI, max:  Math.PI },
        y: { min: -Math.PI, max: Math.PI }
    },
    // Distance bounds for orbital controls
    distance: {
        min: 1.2,
        max: 300
    }
};

// Function to apply camera bounds (copied from main script)
function applyCameraBounds(camera) {
    // Clamp position
    camera.position.x = Math.max(CAMERA_BOUNDS.position.x.min, 
                                Math.min(CAMERA_BOUNDS.position.x.max, camera.position.x));
    camera.position.y = Math.max(CAMERA_BOUNDS.position.y.min, 
                                Math.min(CAMERA_BOUNDS.position.y.max, camera.position.y));
    camera.position.z = Math.max(CAMERA_BOUNDS.position.z.min, 
                                Math.min(CAMERA_BOUNDS.position.z.max, camera.position.z));
    
    // Clamp rotation
    camera.rotation.x = Math.max(CAMERA_BOUNDS.rotation.x.min, 
                                Math.min(CAMERA_BOUNDS.rotation.x.max, camera.rotation.x));
    
    // Normalize Y rotation to stay within bounds
    if (camera.rotation.y > CAMERA_BOUNDS.rotation.y.max) {
        camera.rotation.y = CAMERA_BOUNDS.rotation.y.min + (camera.rotation.y - CAMERA_BOUNDS.rotation.y.max);
    } else if (camera.rotation.y < CAMERA_BOUNDS.rotation.y.min) {
        camera.rotation.y = CAMERA_BOUNDS.rotation.y.max + (camera.rotation.y - CAMERA_BOUNDS.rotation.y.min);
    }
}

// World rotation state variables (for Q/E key functionality)
let isQPressed = false;
let isEPressed = false;

// Function to update world rotation based on Q/E key states (copied from main script)
function updateWorldRotation(deltaTime, worldRotation, splatMesh) {
    const rotationSpeed = 2.0; // Rotation speed in radians per second
    
    if (isQPressed) {
        worldRotation.z += rotationSpeed * deltaTime;
    }
    if (isEPressed) {
        worldRotation.z -= rotationSpeed * deltaTime;
    }
    
    // Apply the rotation to the current mesh
    if (splatMesh) {
        splatMesh.rotation.set(worldRotation.x, worldRotation.y, worldRotation.z);
    }
}

// Function to update controls overlay based on device type (copied from main script)
function updateControlsOverlay(isTouchDevice) {
    const overlay = document.querySelector('.viewer-controls-overlay ul');
    if (!overlay) return;
    
    if (isTouchDevice) {
        // Touch device controls
        overlay.innerHTML = `
            <li>Touch + drag: Pitch & Rotate Scene</li>
            <li>Pinch: Zoom in/out</li>
            <li>Two-finger: Move Camera</li>
        `;
    } else {
        // Desktop controls
        overlay.innerHTML = `
            <li>Click + drag: Pitch & Rotate Scene</li>
            <li>Scroll: Zoom in/out</li>
            <li>WASD: Move Camera</li>
            <li>QERF: Rotate Scene & Up/Down</li>
        `;
    }
}

export const VIEWER_CONFIG = {
    // Performance settings
    performance: {
        antialias: false,
        alpha: true,
        powerPreference: "high-performance",
        maxPixelRatio: 2,
        sortMode: 'stochastic',
        maxPixelRadius: 40,
        minAlpha: 0.02,
        maxStdDev: Math.sqrt(5)
    },
    
    // Camera settings
    camera: {
        fov: 50,
        near: 0.01,
        far: 1000,
        position: { x: 0, y: -10, z: 5}, // Closer position for scene pages
        tiltDegrees: 65 // No tilt for scene pages
    },
    
    // Scene settings
    scene: {
        backgroundColor: 'whitesmoke', // Use same background for both main and scene pages
        mainBackgroundColor: 'whitesmoke', // Light background for main page
        axesHelper: {
            size: 50,
            visible: false
        }
    },
    
    // Controls settings
    controls: {
        moveSpeed: 25,
        QE_KEYCODE_ROTATE: true,
        rotationSpeed: 0.05,
        scrollSpeed: {
            touch: 0.5,
            desktop: 0.15
        },
        slideSpeed: {
            touch: 0.5,
            desktop: 0.006
        }
    },
    
    // World rotation settings
    worldRotation: {
        initial: { x: 0, y: 0, z: 0 },
        speed: 2.0 // Rotation speed in radians per second
    }
};

// Scene-specific splat URLs and metadata
export const SCENE_CONFIG = {
    splats: [
        {
            url: 'assets/splats/scene_52_4_winter.ksplat',
            title: "Scene 1: Winter Scene",
            desc: "Prompt: \"A Residential Neighborhood in the Winter, with Heavy Snow.\"",
            position: { x: 15, y: 18, z: -1 }    // Scene 1: Winter - same as main script
        },
        {
            url: 'assets/splats/scene_75_4_desert.ksplat',
            title: "Scene 2: Desert Scene", 
            desc: "Prompt: \"An Urban Highway in the Desert.\"",
            position: { x: 0, y: 15, z: -1 }     // Scene 2: Desert - same as main script
        },
        {
            url: 'assets/splats/scene_21_3_night.ksplat',
            title: "Scene 3: Night Scene",
            desc: "Prompt: \"A City Street at Night.\"",
            position: { x: 15, y: 15, z: -1 }    // Scene 3: Night - same as main script
        },
        {
            url: 'assets/splats/scene_93_4_suburban.ksplat',
            title: "Scene 4: Urban Scene",
            desc: "Prompt: \"An Urban Neighborhood.\"",
            position: { x: 0, y: 15, z: -1 }     // Scene 4: Urban - same as main script
        }
    ],
    
    previewVideos: [
        'assets/video/high_views_compressed_720p/52_4_highview_rgb.mp4',
        'assets/video/high_views_compressed_720p/75_4_desert_highview_rgb.mp4',
        'assets/video/high_views_compressed_720p/21_3_night_highview_rgb.mp4',
        'assets/video/high_views_compressed_720p/93_4_highview_rgb.mp4'
    ]
};

// Loading indicator functions (copied from main script)
function showLoadingIndicator(container) {
    console.log('Showing loading indicator for unified viewer');
    
    if (container) {
        // Ensure container has relative positioning for absolute overlay
        const computedStyle = window.getComputedStyle(container);
        if (computedStyle.position === 'static') {
            container.style.position = 'relative';
        }
        
        // Also ensure the container has proper dimensions
        if (!container.style.width && !computedStyle.width) {
            container.style.width = '100%';
        }
        if (!container.style.height && !computedStyle.height) {
            container.style.height = '100%';
        }
        
        // Remove any existing overlay
        const existingOverlay = container.querySelector('.viewer-loading-overlay');
        if (existingOverlay) {
            existingOverlay.remove();
        }
        
        // Create new loading overlay
        const overlay = document.createElement('div');
        overlay.className = 'viewer-loading-overlay';
        overlay.style.cssText = `
            position: absolute !important;
            top: 0 !important;
            left: 0 !important;
            right: 0 !important;
            bottom: 0 !important;
            width: 100% !important;
            height: 100% !important;
            background: rgba(34, 34, 34, 0.95) !important;
            display: flex !important;
            flex-direction: column !important;
            align-items: center !important;
            justify-content: center !important;
            z-index: 9999 !important;
            backdrop-filter: blur(2px);
            animation: fadeIn 0.3s ease-in-out;
        `;
        overlay.innerHTML = `
            <div class="viewer-loading-content" style="text-align: center; color: white;">
                <div class="large-loading-spinner" style="
                    border: 3px solid rgba(255, 255, 255, 0.3); 
                    border-top: 3px solid #3498db; 
                    border-radius: 50%; 
                    width: 40px; 
                    height: 40px; 
                    animation: spin 1s linear infinite; 
                    margin: 0 auto 15px auto;
                "></div>
                <div class="loading-progress" style="font-size: 1.1rem; font-weight: 600; margin-bottom: 8px;">Loading</div>
                <div class="loading-subtitle" style="font-size: 0.9rem; opacity: 0.8;">Preparing 3D environment...</div>
            </div>
        `;
        
        // Add the spin animation if not already present
        if (!document.querySelector('#spin-animation-style')) {
            const style = document.createElement('style');
            style.id = 'spin-animation-style';
            style.textContent = `
                @keyframes spin {
                    0% { transform: rotate(0deg); }
                    100% { transform: rotate(360deg); }
                }
            `;
            document.head.appendChild(style);
        }
        container.appendChild(overlay);
    }
}

function updateLoadingIndicator(container, percent) {
    console.log(`Updating loading indicator: ${percent}%`);
    
    if (container) {
        const overlay = container.querySelector('.viewer-loading-overlay');
        if (overlay) {
            const progressElement = overlay.querySelector('.loading-progress');
            const subtitleElement = overlay.querySelector('.loading-subtitle');
            
            if (progressElement) {
                progressElement.textContent = `Loading - ${percent}%`;
            }
            
            if (subtitleElement) {
                if (percent < 30) {
                    subtitleElement.textContent = 'Downloading generated scene...';
                } else if (percent < 70) {
                    subtitleElement.textContent = 'Processing geometry...';
                } else if (percent < 95) {
                    subtitleElement.textContent = 'Finalizing scene...';
                } else {
                    subtitleElement.textContent = 'Almost ready!';
                }
            }
        }
    }
}

function hideLoadingIndicator(container) {
    console.log('Hiding loading indicator');
    
    if (container) {
        const overlay = container.querySelector('.viewer-loading-overlay');
        if (overlay) {
            overlay.style.animation = 'fadeOut 0.3s ease-in-out';
            setTimeout(() => {
                overlay.remove();
            }, 300);
        }
    }
    
    // Remove any old loading indicators
    const indicator = document.querySelector('.loading-indicator');
    if (indicator) {
        indicator.remove();
    }
}

// Unified viewer initialization function
export async function createUnifiedViewer(container, sceneIndex = null, isMainPage = false) {
    const config = VIEWER_CONFIG;
    
    if (!container) {
        throw new Error('Container element not found');
    }
    
    const width = container.offsetWidth || 1200;
    const height = container.offsetHeight || 600;
    
    let viewer, scene, camera, renderer, controls;
    let worldRotation = { ...config.worldRotation.initial };
    let splatMesh = null;
    let axesHelper = null;
    
    try {
        if (isMainPage) {
            // Main page initialization with full THREE.js setup
            scene = new THREE.Scene();
            scene.background = new THREE.Color(config.scene.mainBackgroundColor);
            
            // Add coordinate axes helper
            const axesHelper = new THREE.AxesHelper(config.scene.axesHelper.size);
            axesHelper.visible = config.scene.axesHelper.visible;
            scene.add(axesHelper);
            
            // Create camera
            camera = new THREE.PerspectiveCamera(50, width / height, 0.01, 1000);
            camera.position.set(0, -70, 35);

            // Tilt camera by 15 degrees
            camera.rotation.x = THREE.MathUtils.degToRad(90-25); // Negative for downward tilt
            
            // Create WebGL renderer
            const webglRenderer = new THREE.WebGLRenderer({ 
                antialias: false, // Critical for splat performance
                alpha: true,
                powerPreference: "high-performance"
            });
            webglRenderer.setSize(width, height);

            // Optimize pixel ratio for performance
            webglRenderer.setPixelRatio(Math.min(window.devicePixelRatio, config.performance.maxPixelRatio));
            
            // Create SparkRenderer
            renderer = new SparkRenderer({
                renderer: webglRenderer,
                enable2DGS: true,
                premultipliedAlpha: true,
                maxStdDev: config.performance.maxStdDev
            });
            
            // Apply performance settings
            if (renderer.sortMode !== undefined) renderer.sortMode = config.performance.sortMode;
            if (renderer.maxPixelRadius !== undefined) renderer.maxPixelRadius = config.performance.maxPixelRadius;
            if (renderer.minAlpha !== undefined) renderer.minAlpha = config.performance.minAlpha;
            
            // Style canvas
            webglRenderer.domElement.style.width = '100%';
            webglRenderer.domElement.style.height = '100%';
            webglRenderer.domElement.style.display = 'block';
            webglRenderer.domElement.style.touchAction = 'none';
            
            container.appendChild(webglRenderer.domElement);
            
            return { viewer, scene, camera, renderer, controls, worldRotation, splatMesh, axesHelper };
            
        } else {
            // Scene page initialization with SparkRenderer
            // Try the same pattern as main script - create WebGLRenderer first
            const webglRenderer = new THREE.WebGLRenderer({
                antialias: config.performance.antialias,
                alpha: config.performance.alpha,
                powerPreference: config.performance.powerPreference
            });
            
            webglRenderer.setSize(container.clientWidth, container.clientHeight);
            webglRenderer.setPixelRatio(Math.min(window.devicePixelRatio, config.performance.maxPixelRatio));
            
            // Create SparkRenderer with configuration object
            viewer = new SparkRenderer({
                renderer: webglRenderer,
                enable2DGS: true,
                premultipliedAlpha: true,
                maxStdDev: config.performance.maxStdDev
            });
            
            // Create scene and camera manually like in main script
            scene = new THREE.Scene();

            // Create camera
            camera = new THREE.PerspectiveCamera(50, width / height, 0.01, 1000);
            camera.position.set(0, -70, 35);
            
            // Tilt camera by 15 degrees
            camera.rotation.x = THREE.MathUtils.degToRad(90-25); // Negative for downward tilt
            
            // No rotation/tilt for scene pages - let users control with SparkControls
            
            // Configure scene
            scene.background = new THREE.Color(config.scene.backgroundColor);
            
            // Add axes helper if needed
            if (config.scene.axesHelper.size > 0) {
                axesHelper = new THREE.AxesHelper(config.scene.axesHelper.size);
                axesHelper.visible = config.scene.axesHelper.visible;
                scene.add(axesHelper);
            }
            
            // Style and append canvas first
            webglRenderer.domElement.style.width = '100%';
            webglRenderer.domElement.style.height = '100%';
            webglRenderer.domElement.style.display = 'block';
            webglRenderer.domElement.style.touchAction = 'none';
            
            container.appendChild(webglRenderer.domElement);
            
            // Create controls after canvas is in DOM - use same pattern as main script
            controls = new SparkControls({ 
                canvas: webglRenderer.domElement
            });
            controls.fpsMovement.moveSpeed = config.controls.moveSpeed;
            
            // Detect touch devices and increase scroll speed for better mobile experience
            const isTouchDevice = 'ontouchstart' in window || navigator.maxTouchPoints > 0;
            console.log('Touch device detection:', {
                isTouchDevice: isTouchDevice,
                ontouchstart: 'ontouchstart' in window,
                maxTouchPoints: navigator.maxTouchPoints,
                userAgent: navigator.userAgent
            });
            
            controls.pointerControls.scrollSpeed = isTouchDevice ? 0.5 : 0.15; // Higher speed for touch devices
            controls.pointerControls.slideSpeed = isTouchDevice ? 0.5 : 0.006; // Higher speed for touch devices
            console.log('Scroll speed set to:', controls.pointerControls.scrollSpeed, '(Touch device:', isTouchDevice + ')');
            
            controls.pointerControls.pointerRollScale = 0; // Disable pointer rotation
            controls.fpsMovement.rollSpeed = 0; // Disable camera roll with Q and E
            // Z-axis rotation is handled by the custom update function below
            
            // Store the original update function
            const originalPointerUpdate = controls.pointerControls.update.bind(controls.pointerControls);
            
            // Overwrite the pointer controls update function to convert YZ camera rotation to world rotation
            controls.pointerControls.update = function(deltaTime, control) {
                // Store the original YZ rotation before update
                const originalYRotation = control.rotation.y;
                const originalZRotation = control.rotation.z;
                
                // Call the original update function
                originalPointerUpdate(deltaTime, control);
                
                // Calculate the rotation deltas that would have been applied to camera
                const deltaY = control.rotation.y - originalYRotation;
                const deltaZ = control.rotation.z - originalZRotation;
                
                // Apply Y and Z rotation deltas to world rotation instead of camera
                // if (Math.abs(deltaY) > 0.0001) { // Only apply if there's meaningful rotation
                //     worldRotation.y += deltaY;
                //     if (splatMesh) {
                //         splatMesh.rotation.set(worldRotation.x, worldRotation.y, worldRotation.z);
                //     }
                // }
                if (Math.abs(deltaZ) > 0.0001) { // Only apply if there's meaningful rotation
                    worldRotation.z += deltaZ;
                    if (splatMesh) {
                        splatMesh.rotation.set(worldRotation.x, worldRotation.y, worldRotation.z);
                    }
                }
                
                // Restore the YZ rotation to prevent camera rotation
                control.rotation.y = originalYRotation;
                control.rotation.z = originalZRotation;
            };
            
            // Apply bounds to SparkControls if available (these are for different control types)
            if (controls.minDistance !== undefined) controls.minDistance = CAMERA_BOUNDS.distance.min;
            if (controls.maxDistance !== undefined) controls.maxDistance = CAMERA_BOUNDS.distance.max;
            if (controls.minPolarAngle !== undefined) controls.minPolarAngle = 0;
            if (controls.maxPolarAngle !== undefined) controls.maxPolarAngle = Math.PI;
            
            // Load the specific scene splat
            if (sceneIndex !== null && SCENE_CONFIG.splats[sceneIndex]) {
                const splatConfig = SCENE_CONFIG.splats[sceneIndex];
                
                // Show loading indicator
                showLoadingIndicator(container);
                
                try {
                    // Use SplatLoader to load the splat data directly from URL
                    const loader = new SplatLoader();
                    const packedSplats = await loader.loadAsync(splatConfig.url, (progress) => {
                        if (progress.lengthComputable) {
                            const percent = Math.min(100, Math.round((progress.loaded / progress.total) * 100));
                            console.log(`Loading unified viewer splat: ${percent}%`);
                            updateLoadingIndicator(container, percent);
                        }
                    });
                    
                    // Create SplatMesh from the loaded data
                    splatMesh = new SplatMesh({ packedSplats: packedSplats });
                    
                    // Set position from config
                    const position = splatConfig.position || { x: 0, y: 0, z: -3 }; // Fallback to match main script
                    splatMesh.position.set(position.x, position.y, position.z);
                    
                    // Enable matrix auto updates for rotation
                    splatMesh.matrixAutoUpdate = true;
                    
                    console.log('Splat loaded and added:', {
                        splatMesh,
                        position: splatMesh.position,
                        camera: camera.position,
                        scene: scene
                    });
                    
                    scene.add(splatMesh);
                    
                    // Hide loading indicator after successful load
                    hideLoadingIndicator(container);
                    
                } catch (error) {
                    console.error('Error loading splat in unified viewer:', error);
                    hideLoadingIndicator(container);
                }
            }
            
            // Start the render loop
            let lastTime = 0;
            webglRenderer.setAnimationLoop(function animate(time) {
                const deltaTime = (time - (lastTime || time)) / 1000;
                lastTime = time;
                
                // Update controls
                if (controls) {
                    controls.update(camera);
                }
                
                // Update world rotation based on Q/E key states
                updateWorldRotation(deltaTime, worldRotation, splatMesh);
                
                // Apply camera bounds after control updates
                applyCameraBounds(camera);
                
                // Render the scene
                webglRenderer.render(scene, camera);
            });
            
            return { viewer, scene, camera, renderer: viewer, controls, worldRotation, splatMesh, axesHelper };
        }
        
    } catch (error) {
        console.error('Error creating unified viewer:', error);
        throw error;
    }
}

// Unified controls setup function
export function setupUnifiedControls(viewerComponents) {
    const { viewer, scene, camera, controls, worldRotation, splatMesh } = viewerComponents;
    const config = VIEWER_CONFIG;
    
    // Initialize isTouchDevice detection and update controls overlay (copied from main script)
    const isTouchDevice = 'ontouchstart' in window || navigator.maxTouchPoints > 0;
    console.log('Touch device detection in setupUnifiedControls:', {
        isTouchDevice: isTouchDevice,
        ontouchstart: 'ontouchstart' in window,
        maxTouchPoints: navigator.maxTouchPoints,
        userAgent: navigator.userAgent
    });
    updateControlsOverlay(isTouchDevice);
    
    // Setup Q/E keyboard controls for world rotation
    document.addEventListener('keydown', (event) => {
        switch(event.code) {
            case 'KeyQ':
                isQPressed = true;
                event.preventDefault();
                break;
            case 'KeyE':
                isEPressed = true;
                event.preventDefault();
                break;
        }
    });
    
    document.addEventListener('keyup', (event) => {
        switch(event.code) {
            case 'KeyQ':
                isQPressed = false;
                break;
            case 'KeyE':
                isEPressed = false;
                break;
        }
    });
    
    // Setup X key to toggle axes visibility (copied from main script)
    document.addEventListener('keydown', (event) => {
        if (event.key === 'x' || event.key === 'X') {
            const { axesHelper } = viewerComponents;
            if (axesHelper) {
                axesHelper.visible = !axesHelper.visible;
                console.log('Coordinate axes', axesHelper.visible ? 'shown' : 'hidden');
            }
        }
    });
    
    // Setup pointer controls override for world rotation (scene pages only)
    if (viewer && viewer.controls && viewer.controls.pointerControls) {
        const originalUpdate = viewer.controls.pointerControls.update;
        viewer.controls.pointerControls.update = function() {
            originalUpdate.call(this);
            
            if (this.rotationDelta && splatMesh) {
                if (Math.abs(this.rotationDelta.y) > 0.001) {
                    worldRotation.y += this.rotationDelta.y;
                }
                if (Math.abs(this.rotationDelta.z) > 0.001) {
                    worldRotation.x -= this.rotationDelta.z;
                }
                
                // Apply world rotation to splat mesh
                if (splatMesh) {
                    splatMesh.rotation.set(worldRotation.x, worldRotation.y, worldRotation.z);
                }
                
                this.rotationDelta.y = 0;
                this.rotationDelta.z = 0;
            }
        };
    }
    
    return { viewerComponents };
}
