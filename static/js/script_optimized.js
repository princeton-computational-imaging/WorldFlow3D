import * as THREE from "three";
import { SparkControls, SplatMesh, SparkRenderer, SplatLoader } from "@sparkjsdev/spark";
import { VIEWER_CONFIG, SCENE_CONFIG, createUnifiedViewer, setupUnifiedControls } from './unified-viewer-config.js';

let scene, camera, renderer, sparkControls;
let currentSlideIndex = 0;
let splats = []; // Store loaded splat data
let splatMeshes = []; // Store actual mesh instances
let isTransitioning = false;
let loadingStates = [false, false, false, false]; // Track loading state of each splat
let axesHelper; // Coordinate axes helper for debugging

const splatUrls = [
    'assets/splats/scene_52_4_winter.ksplat',
    'assets/splats/scene_75_4_desert.ksplat',
    'assets/splats/scene_21_3_night.ksplat',
    'assets/splats/scene_93_4_suburban.ksplat'
];

const slideData = [
    { title: "Scene 1: Winter Scene", desc: "Prompt: \"A Residential Neighborhood in the Winter, with Heavy Snow.\" " },
    { title: "Scene 2: Desert Scene", desc: "Prompt: \"An Urban Highway in the Desert.\" " },
    { title: "Scene 3: Night Scene", desc: "Prompt: \"A City Street at Night.\" " },
    { title: "Scene 4: Urban Scene", desc: "Prompt: \"An Urban Neighborhood.\" " },

];

// Position configuration for each splat scene
const splatPositions = [
    { x: 15, y: 18, z: -1 },    // Scene 1: Winter - default position
    { x: 0, y: 15, z: -1 },     // Scene 2: Desert - default position
    { x: 15, y: 15, z: -1 },    // Scene 3: Night - closer to camera
    { x: 0, y: 15, z: -1 },    // Scene 4: Urban - further from camera

];

// Video mapping for preview panels
const previewVideos = [
    // 'assets/video/high_views_compressed_720p/52_4_highview_rgb.mp4',     // Scene 1: Winter
    'assets/video/high_views_compressed_720p/11_2_winter_highview_rgb.mp4',     // Scene 1: Winter
    'assets/video/high_views_compressed_720p/75_4_desert_highview_rgb.mp4',  // Scene 2: Desert
    'assets/video/high_views_compressed_720p/21_3_night_highview_rgb.mp4', // Scene 3: Night
    'assets/video/high_views_compressed_720p/93_4_highview_rgb.mp4',     // Scene 4: Urban
];

// Camera bounds configuration
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

// World rotation state
let worldRotation = { x: 0, y: 0, z: 0 };
let isQPressed = false;
let isEPressed = false;

// Function to detect Chrome or Edge on Windows for fallback
function isChromeOrEdgeOnWindows() {
    const isChrome = /Chrome/.test(navigator.userAgent) && /Google Inc/.test(navigator.vendor);
    const isEdge = /Edg/.test(navigator.userAgent); // Modern Edge uses 'Edg' in user agent
    const isWindows = /Windows/.test(navigator.userAgent);
    return (isChrome || isEdge) && isWindows;
    // return isChrome;
}

// Function to show video fallback for Chrome or Edge on Windows
function showVideoFallback(container, sceneIndex) {
    const videoSrc = previewVideos[sceneIndex];
    
    container.innerHTML = `
        <div class="video-fallback-container" style="position: relative; width: 100%; height: 100%; display: flex; align-items: center; justify-content: center;">
            <video class="fallback-video" muted loop playsinline autoplay style="width: 100%; height: 100%; object-fit: cover;">
                <source src="${videoSrc}" type="video/mp4">
                Your browser does not support the video tag.
            </video>
            <div class="video-fallback-overlay" style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; display: flex; align-items: center; justify-content: center;">
                <button class="button is-primary is-rounded fallback-button" onclick="openInteractiveScene(${sceneIndex})" style="background: #ff8f00; border: none; padding: 12px 24px; border-radius: 25px; color: white; font-size: 1rem; cursor: pointer; transition: all 0.3s ease; display: inline-flex; align-items: center; gap: 8px; box-shadow: 0 4px 15px rgba(173, 93, 36, 0.3);" onmouseover="this.style.background='#ff8f00'; this.style.transform='translateY(-2px)'; this.style.boxShadow='0 6px 20px rgba(173, 93, 36, 0.4)';" onmouseout="this.style.background='#ff8f00'; this.style.transform='translateY(0)'; this.style.boxShadow='0 4px 15px rgba(173, 93, 36, 0.3)';">
                    <span class="icon">
                        <i class="fas fa-cube"></i>
                    </span>
                    <span>View Interactive Scene</span>
                </button>
            </div>
        </div>
    `;
    
    // Hide the controls overlay for Chrome/Edge on Windows fallback
    const controlsOverlay = document.querySelector('.viewer-controls-overlay');
    if (controlsOverlay) {
        controlsOverlay.style.display = 'none';
    }
}

// Function to open interactive scene in same window
function openInteractiveScene(sceneIndex) {
    const scenePages = [
        'scene1.html',
        'scene2.html', 
        'scene3.html',
        'scene4.html'
    ];
    window.location.href = scenePages[sceneIndex];
}

// Make the function globally accessible for inline onclick handlers
window.openInteractiveScene = openInteractiveScene;

// Function to apply camera bounds
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

// Function to toggle coordinate axes visibility
function toggleAxes() {
    if (axesHelper) {
        axesHelper.visible = !axesHelper.visible;
        console.log('Coordinate axes', axesHelper.visible ? 'shown' : 'hidden');
    }
}

// Function to apply world rotation to current splat mesh
function applyWorldRotation() {
    const currentMesh = splatMeshes[currentSlideIndex];
    if (currentMesh) {
        currentMesh.rotation.set(worldRotation.x, worldRotation.y, worldRotation.z);
    }
}

// Function to update world rotation based on Q/E key states
function updateWorldRotation(deltaTime) {
    const rotationSpeed = 2.0; // Rotation speed in radians per second
    
    if (isQPressed) {
        worldRotation.z += rotationSpeed * deltaTime;
    }
    if (isEPressed) {
        worldRotation.z -= rotationSpeed * deltaTime;
    }
    
    // Apply the rotation to the current mesh
    applyWorldRotation();
}

// Function to update controls overlay based on device type
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

// Performance optimized viewer initialization
function initViewer() {
    const container = document.getElementById('viewer-active');
    if (!container) {
        console.error('Container viewer-active not found');
        return;
    }
    
    console.log('Initializing optimized viewer...');
    
    // Check if Chrome or Edge on Windows - use video fallback
    if (isChromeOrEdgeOnWindows()) {
        console.log('Chrome or Edge on Windows detected - using video fallback');
        showVideoFallback(container, currentSlideIndex);
        return;
    }
    
    const width = container.offsetWidth || 1200;
    const height = container.offsetHeight || 600;
    
    // Create scene
    scene = new THREE.Scene();
    scene.background = new THREE.Color('whitesmoke');

    // Add coordinate axes helper for orientation reference
    // Red = X-axis, Green = Y-axis, Blue = Z-axis
    axesHelper = new THREE.AxesHelper(50); // Size of 50 units
    axesHelper.visible = false; // Hidden by default, toggle with X key
    scene.add(axesHelper);
    
    // Create camera
    camera = new THREE.PerspectiveCamera(50, width / height, 0.01, 1000);
    camera.position.set(0, -70, 35);
    
    // Tilt camera by 15 degrees
    camera.rotation.x = THREE.MathUtils.degToRad(90-25); // Negative for downward tilt

    // Create WebGL renderer with performance optimizations
    const webglRenderer = new THREE.WebGLRenderer({ 
        antialias: false, // Critical for splat performance
        alpha: true,
        powerPreference: "high-performance"
    });
    webglRenderer.setSize(width, height);
    
    // Optimize pixel ratio for performance
    const pixelRatio = Math.min(window.devicePixelRatio, 2);
    webglRenderer.setPixelRatio(pixelRatio);
    
    // Create SparkRenderer with performance optimizations
    renderer = new SparkRenderer({ 
        renderer: webglRenderer,
        enable2DGS: true,
        premultipliedAlpha: true,
        maxStdDev: Math.sqrt(5) // Optimized for performance
    });
    
    // Apply performance settings
    if (renderer.sortMode !== undefined) renderer.sortMode = 'stochastic';
    if (renderer.maxPixelRadius !== undefined) renderer.maxPixelRadius = 40; // Reduced from 50
    if (renderer.minAlpha !== undefined) renderer.minAlpha = 0.02; // Slightly higher for performance
    // if (renderer.maxStdDev !== undefined) renderer.maxStdDev = Math.sqrt(5); // Optimized for performance
    
    // Style canvas
    webglRenderer.domElement.style.width = '100%';
    webglRenderer.domElement.style.height = '100%';
    webglRenderer.domElement.style.display = 'block';
    webglRenderer.domElement.style.touchAction = 'none';
    
    container.appendChild(webglRenderer.domElement);
    
    // Setup optimized controls
    try {
        sparkControls = new SparkControls({ 
            canvas: webglRenderer.domElement
        });
        sparkControls.fpsMovement.moveSpeed = 25; // Increase movement speed
        
        // Detect touch devices and increase scroll speed for better mobile experience
        const isTouchDevice = 'ontouchstart' in window || navigator.maxTouchPoints > 0;
        console.log('Touch device detection:', {
            isTouchDevice: isTouchDevice,
            ontouchstart: 'ontouchstart' in window,
            maxTouchPoints: navigator.maxTouchPoints,
            userAgent: navigator.userAgent
        });
        
        sparkControls.pointerControls.scrollSpeed = isTouchDevice ? 0.5 : 0.15; // Higher speed for touch devices
        sparkControls.pointerControls.slideSpeed = isTouchDevice ? 0.5 : 0.006; // Higher speed for touch devices
        console.log('Scroll speed set to:', sparkControls.pointerControls.scrollSpeed, '(Touch device:', isTouchDevice + ')');
        
        // Update viewer controls overlay based on device type
        updateControlsOverlay(isTouchDevice);
        
        sparkControls.pointerControls.pointerRollScale = 0; // Disable pointer rotation
        sparkControls.fpsMovement.rollSpeed = 0; // Disable camera roll with Q and E
        // Z-axis rotation is handled by the custom update function below

        // Store the original update function
        const originalPointerUpdate = sparkControls.pointerControls.update.bind(sparkControls.pointerControls);
        
        // Overwrite the pointer controls update function to convert YZ camera rotation to world rotation
        sparkControls.pointerControls.update = function(deltaTime, control) {
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
            //     applyWorldRotation();
            // }
            if (Math.abs(deltaZ) > 0.0001) { // Only apply if there's meaningful rotation
                worldRotation.z += deltaZ;
                applyWorldRotation();
            }
            
            // Restore the YZ rotation to prevent camera rotation
            control.rotation.y = originalYRotation;
            control.rotation.z = originalZRotation;
        };

        // Apply bounds to SparkControls if available (these are for different control types)
        if (sparkControls.minDistance !== undefined) sparkControls.minDistance = CAMERA_BOUNDS.distance.min;
        if (sparkControls.maxDistance !== undefined) sparkControls.maxDistance = CAMERA_BOUNDS.distance.max;
        if (sparkControls.minPolarAngle !== undefined) sparkControls.minPolarAngle = 0;
        if (sparkControls.maxPolarAngle !== undefined) sparkControls.maxPolarAngle = Math.PI;
        
        console.log('SparkControls initialized successfully with bounds');
    } catch (error) {
        console.error('Failed to initialize SparkControls:', error);
        sparkControls = setupFallbackControls(webglRenderer.domElement, camera);
    }
    
    // Initialize splat arrays
    splats = new Array(splatUrls.length).fill(null);
    splatMeshes = new Array(splatUrls.length).fill(null);
    
    // Show loading indicator and load first splat immediately
    showLoadingIndicator(0);
    loadSplatAsync(0, true);
    
    // Start optimized animation loop
    startOptimizedAnimation();
    
    // Handle window resize
    window.addEventListener('resize', handleResize);
    
    console.log('Optimized viewer initialization complete');
}

// Async splat loading with progress tracking
async function loadSplatAsync(index, addToScene = false) {
    if (loadingStates[index] || splats[index]) {
        console.log(`Splat ${index} already loaded or loading`);
        return;
    }
    
    console.log(`Loading splat ${index}: ${splatUrls[index]}`);
    loadingStates[index] = true;
    
    try {
        const loader = new SplatLoader();
        
        // Load with progress tracking
        const packedSplats = await loader.loadAsync(splatUrls[index], (progress) => {
            if (progress.lengthComputable) {
                const percent = Math.min(100, Math.round((progress.loaded / progress.total) * 100));
                console.log(`Loading splat ${index}: ${percent}%`);
                updateLoadingIndicator(index, percent);
            }
        });
        
        // Store the packed splats data
        splats[index] = packedSplats;
        
        // Create mesh only when needed
        if (addToScene || index === currentSlideIndex) {
            createAndAddSplatMesh(index);
        }
        
        console.log(`Splat ${index} loaded successfully`);
        
    } catch (error) {
        console.error(`Error loading splat ${index}:`, error);
    } finally {
        loadingStates[index] = false;
        hideLoadingIndicator(index);
    }
}

// Create mesh from loaded splat data
function createAndAddSplatMesh(index) {
    if (!splats[index] || splatMeshes[index]) {
        return;
    }
    
    const splatMesh = new SplatMesh({ packedSplats: splats[index] });
    
    // Use position from splatPositions array for this specific scene
    const position = splatPositions[index] || { x: 0, y: 0, z: -3 }; // Fallback to default
    splatMesh.position.set(position.x, position.y, position.z);
    
    // Apply current world rotation to the new mesh
    splatMesh.rotation.set(worldRotation.x, worldRotation.y, worldRotation.z);
    
    // Keep matrix auto updates enabled for rotation
    splatMesh.scale.setScalar(1);
    splatMesh.matrixAutoUpdate = true; // Enable for rotation updates
    
    splatMeshes[index] = splatMesh;
    
    if (index === currentSlideIndex) {
        scene.add(splatMesh);
    }
}

// Optimized splat switching with lazy loading
async function switchToSplat(newIndex) {
    if (isTransitioning || newIndex === currentSlideIndex) {
        return;
    }
    
    console.log(`Switching from splat ${currentSlideIndex} to ${newIndex}`);
    isTransitioning = true;
    
    // Update currentSlideIndex early so loading indicators work correctly
    const previousSlideIndex = currentSlideIndex;
    currentSlideIndex = newIndex;
    
    // If Chrome on Windows, just update the video fallback and skip splat loading
    if (isChromeOrEdgeOnWindows()) {
        setTimeout(() => {
            isTransitioning = false;
            console.log(`Switched to video fallback for scene ${newIndex}`);
        }, 100);
        return;
    }
    
    // Remove current splat from scene
    if (splatMeshes[previousSlideIndex]) {
        scene.remove(splatMeshes[previousSlideIndex]);
    }
    
    // Check if we need to load the new splat and show loading indicator if needed
    if (!splats[newIndex]) {
        showLoadingIndicator(newIndex);
        await loadSplatAsync(newIndex, true);
    } else if (!splatMeshes[newIndex]) {
        createAndAddSplatMesh(newIndex);
    }
    
    // Add new splat to scene
    if (splatMeshes[newIndex]) {
        scene.add(splatMeshes[newIndex]);
        // Apply current world rotation to the new mesh
        applyWorldRotation();
    }
    
    // Preload adjacent splats for smooth navigation
    preloadAdjacentSplats(newIndex);
    
    // Clean up unused splats to free memory
    cleanupUnusedSplats(newIndex);
    
    setTimeout(() => {
        isTransitioning = false;
        console.log(`Switched to splat ${newIndex}`);
    }, 100);
}

// Preload splats adjacent to current one
function preloadAdjacentSplats(currentIndex) {
    // Skip preloading on Chrome/Windows since we're using video fallback
    if (isChromeOrEdgeOnWindows()) {
        return;
    }
    
    const prev = (currentIndex - 1 + splatUrls.length) % splatUrls.length;
    const next = (currentIndex + 1) % splatUrls.length;
    
    // Load adjacent splats in the background
    if (!splats[prev]) {
        loadSplatAsync(prev);
    }
    if (!splats[next]) {
        loadSplatAsync(next);
    }
}

// Clean up unused splats to free memory
function cleanupUnusedSplats(currentIndex) {
    const keep = new Set([
        currentIndex,
        (currentIndex - 1 + splatUrls.length) % splatUrls.length,
        (currentIndex + 1) % splatUrls.length
    ]);
    
    for (let i = 0; i < splatMeshes.length; i++) {
        if (!keep.has(i) && splatMeshes[i]) {
            // Dispose of mesh and clear references
            if (splatMeshes[i].parent) {
                scene.remove(splatMeshes[i]);
            }
            splatMeshes[i].dispose?.();
            splatMeshes[i] = null;
            
            // Keep splat data but clear mesh
            console.log(`Cleaned up splat mesh ${i} to free memory`);
        }
    }
}

// Optimized animation loop
function startOptimizedAnimation() {
    let lastTime = 0;
    const webglRenderer = renderer.renderer || renderer;
    
    webglRenderer.setAnimationLoop(function animate(time) {
        const deltaTime = (time - (lastTime || time)) / 1000; // Convert to seconds
        lastTime = time;
        
        // Update controls with damping
        if (sparkControls) {
            try {
                sparkControls.update(camera);
            } catch (error) {
                console.error('SparkControls error:', error);
            }
        }
        
        // Update world rotation based on Q/E key states
        updateWorldRotation(deltaTime);
        
        // Always apply camera bounds after control updates
        applyCameraBounds(camera);
        
        // Use WebGLRenderer render method (SparkRenderer doesn't have render method)
        webglRenderer.render(scene, camera);
    });
}

// Fallback controls if SparkControls fails
function setupFallbackControls(domElement, camera) {
    let isMouseDown = false;
    let mouseX = 0, mouseY = 0;
    let targetRotationX = -1.309, targetRotationY = 0;
    let rotationX = -1.309, rotationY = 0;
    
    domElement.addEventListener('mousedown', (event) => {
        isMouseDown = true;
        mouseX = event.clientX;
        mouseY = event.clientY;
    });
    
    domElement.addEventListener('mousemove', (event) => {
        if (!isMouseDown) return;
        
        const deltaX = event.clientX - mouseX;
        const deltaY = event.clientY - mouseY;
        
        targetRotationY += deltaX * 0.01;
        targetRotationX += deltaY * 0.01;
        
        // Apply rotation bounds
        targetRotationX = Math.max(CAMERA_BOUNDS.rotation.x.min / Math.PI, 
                                  Math.min(CAMERA_BOUNDS.rotation.x.max / Math.PI, targetRotationX));
        
        mouseX = event.clientX;
        mouseY = event.clientY;
    });
    
    domElement.addEventListener('mouseup', () => {
        isMouseDown = false;
    });
    
    domElement.addEventListener('wheel', (event) => {
        event.preventDefault();
        camera.position.z += event.deltaY * 0.01;
        // Apply distance bounds
        camera.position.z = Math.max(CAMERA_BOUNDS.distance.min, 
                                    Math.min(CAMERA_BOUNDS.distance.max, camera.position.z));
    });
    
    return {
        update: () => {
            rotationX += (targetRotationX - rotationX) * 0.05;
            rotationY += (targetRotationY - rotationY) * 0.05;
            
            const radius = camera.position.z;
            camera.position.x = Math.sin(rotationY) * radius;
            camera.position.y = Math.sin(rotationX) * radius;
            
            // Apply bounds to final position
            applyCameraBounds(camera);
            
            camera.lookAt(0, 0, 0);
        }
    };
}

// Loading indicator functions
function showLoadingIndicator(index) {
    console.log(`Showing loading indicator for scene ${index}`);
    
    // Show viewer loading overlay - always show for the active viewer since we're switching to this scene
    const viewerContainer = document.getElementById('viewer-active');
    if (viewerContainer) {
        // Remove any existing overlay
        const existingOverlay = viewerContainer.querySelector('.viewer-loading-overlay');
        if (existingOverlay) {
            existingOverlay.remove();
        }
        
        // Create new loading overlay
        const overlay = document.createElement('div');
        overlay.className = 'viewer-loading-overlay';
        overlay.innerHTML = `
            <div class="viewer-loading-content">
                <div class="large-loading-spinner"></div>
                <div class="loading-progress">Loading</div>
                <div class="loading-subtitle">Preparing 3D environment...</div>
            </div>
        `;
        viewerContainer.appendChild(overlay);
    }
}

function updateLoadingIndicator(index, percent) {
    console.log(`Updating loading indicator for scene ${index}: ${percent}%`);
    
    // Update viewer overlay progress - update the currently visible overlay
    const viewerContainer = document.getElementById('viewer-active');
    if (viewerContainer) {
        const overlay = viewerContainer.querySelector('.viewer-loading-overlay');
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

function hideLoadingIndicator(index) {
    console.log(`Hiding loading indicator for scene ${index}`);
    
    // // Hide loading status in slide info
    // const loadingStatus = document.querySelector('.loading-status');
    // if (loadingStatus) {
    //     loadingStatus.style.display = 'none';
    // }
    
    // Remove viewer loading overlay with fade out animation
    const viewerContainer = document.getElementById('viewer-active');
    if (viewerContainer) {
        const overlay = viewerContainer.querySelector('.viewer-loading-overlay');
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

// Optimized slide change function
function changeSlide(direction) {
    if (isTransitioning) return;
    
    console.log(`changeSlide called with direction: ${direction}`);
    
    const dots = document.querySelectorAll('.dot');
    
    // Update dot indicators
    if (dots[currentSlideIndex]) {
        dots[currentSlideIndex].classList.remove('active');
    }
    
    // Calculate new index
    let newIndex = currentSlideIndex + direction;
    if (newIndex >= splatUrls.length) newIndex = 0;
    if (newIndex < 0) newIndex = splatUrls.length - 1;
    
    // Update carousel visual state
    updateCarouselState(newIndex);
    
    // Update dot indicators
    if (dots[newIndex]) {
        dots[newIndex].classList.add('active');
    }
    
    // Switch splats with lazy loading
    switchToSplat(newIndex);
}

function updateCarouselState(activeIndex) {
    // Calculate prev and next indices
    const prevIndex = (activeIndex - 1 + splatUrls.length) % splatUrls.length;
    const nextIndex = (activeIndex + 1) % splatUrls.length;
    
    // Update active slide info
    const activeSlideInfo = document.querySelector('.active-slide .slide-info h4');
    const activeSlideDesc = document.querySelector('.active-slide .slide-info h3');
    if (activeSlideInfo) activeSlideInfo.textContent = slideData[activeIndex].title;
    if (activeSlideDesc) activeSlideDesc.textContent = slideData[activeIndex].desc;
    
    // If Chrome on Windows, update the video fallback in the center
    if (isChromeOrEdgeOnWindows()) {
        const container = document.getElementById('viewer-active');
        if (container) {
            showVideoFallback(container, activeIndex);
        }
    }
    
    // Update prev slide preview
    const prevSlideTitle = document.querySelector('.prev-slide .preview-overlay h4');
    const prevSlideVideo = document.querySelector('.prev-slide .preview-video source');
    if (prevSlideTitle) prevSlideTitle.textContent = slideData[prevIndex].title;
    if (prevSlideVideo) {
        prevSlideVideo.src = previewVideos[prevIndex];
        const video = prevSlideVideo.parentElement;
        video.load(); // Reload video with new source
        video.play().catch(e => console.warn('Video play failed:', e));
    }
    
    // Update next slide preview
    const nextSlideTitle = document.querySelector('.next-slide .preview-overlay h4');
    const nextSlideVideo = document.querySelector('.next-slide .preview-video source');
    if (nextSlideTitle) nextSlideTitle.textContent = slideData[nextIndex].title;
    if (nextSlideVideo) {
        nextSlideVideo.src = previewVideos[nextIndex];
        const video = nextSlideVideo.parentElement;
        video.load(); // Reload video with new source
        video.play().catch(e => console.warn('Video play failed:', e));
    }
}

// Global function for dot navigation
window.currentSlide = function(n) {
    if (isTransitioning) return;
    
    const dots = document.querySelectorAll('.dot');
    dots[currentSlideIndex].classList.remove('active');
    
    const newIndex = n - 1;
    updateCarouselState(newIndex);
    dots[newIndex].classList.add('active');
    
    switchToSplat(newIndex);
};

// Optimized resize handler
function handleResize() {
    const container = document.getElementById('viewer-active');
    if (!container) return;
    
    const width = container.offsetWidth;
    const height = container.offsetHeight;
    
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    
    const webglRenderer = renderer.renderer || renderer;
    if (webglRenderer && webglRenderer.setSize) {
        webglRenderer.setSize(width, height, false);
    }
}

// Initialize carousel with optimizations
function initCarousel() {
    console.log('Initializing optimized carousel...');
    
    try {
        initViewer();
        updateCarouselState(0);
        
        // Setup event listeners
        setTimeout(() => {
            const prevBtn = document.getElementById('prevBtn');
            const nextBtn = document.getElementById('nextBtn');
            
            if (prevBtn) {
                prevBtn.addEventListener('click', (e) => {
                    e.preventDefault();
                    changeSlide(-1);
                });
            }
            
            if (nextBtn) {
                nextBtn.addEventListener('click', (e) => {
                    e.preventDefault();
                    changeSlide(1);
                });
            }
            
            // Add keyboard navigation
            document.addEventListener('keydown', (e) => {
                // if (e.key === 'ArrowLeft') changeSlide(-1);
                // if (e.key === 'ArrowRight') changeSlide(1);
                if (e.key === 'x' || e.key === 'X') toggleAxes(); // Toggle axes with X key
                
                // Handle Q/E for world rotation
                if (e.code === 'KeyQ') {
                    isQPressed = true;
                    e.preventDefault(); // Prevent other Q key behaviors
                }
                if (e.code === 'KeyE') {
                    isEPressed = true;
                    e.preventDefault(); // Prevent other E key behaviors
                }
            });
            
            // Add keyup handler to stop rotation
            document.addEventListener('keyup', (e) => {
                if (e.code === 'KeyQ') {
                    isQPressed = false;
                }
                if (e.code === 'KeyE') {
                    isEPressed = false;
                }
            });
            
            console.log('Optimized carousel initialized successfully');
        }, 100);
        
    } catch (error) {
        console.error('Error during optimized carousel initialization:', error);
    }
}

// Cleanup function for memory management
function cleanup() {
    if (renderer) {
        const webglRenderer = renderer.renderer || renderer;
        webglRenderer.setAnimationLoop(null);
    }
    
    splatMeshes.forEach(mesh => {
        if (mesh) {
            if (mesh.parent) scene.remove(mesh);
            mesh.dispose?.();
        }
    });
    
    splats = [];
    splatMeshes = [];
}

// Video Marquee Functionality - Full working version with all videos
function initVideoMarquee() {
    console.log('Initializing video marquee...');
    const marqueeTrack = document.getElementById('videoMarqueeTrack');
    if (!marqueeTrack) {
        console.error('Video marquee track not found!');
        return;
    }

    // Clear existing content
    marqueeTrack.innerHTML = '';
    
    // All available high-view videos
    const videoSources = [
        'assets/video/high_views_compressed_720p/5_2_highview_rgb.mp4',
        'assets/video/high_views_compressed_720p/7_4_highview_rgb.mp4',
        'assets/video/high_views_compressed_720p/11_2_winter_highview_rgb.mp4',
        'assets/video/high_views_compressed_720p/17_4_highview_rgb.mp4',
        'assets/video/high_views_compressed_720p/18_3_highview_rgb.mp4',
        'assets/video/high_views_compressed_720p/21_3_night_highview_rgb.mp4',
        'assets/video/high_views_compressed_720p/29_3_highview_rgb.mp4',
        'assets/video/high_views_compressed_720p/37_4_highview_rgb.mp4',
        'assets/video/high_views_compressed_720p/52_4_highview_rgb.mp4',
        'assets/video/high_views_compressed_720p/53_4_highview_rgb.mp4',
        'assets/video/high_views_compressed_720p/75_4_desert_highview_rgb.mp4',
        'assets/video/high_views_compressed_720p/93_4_highview_rgb.mp4',
        'assets/video/high_views_compressed_720p/95_3_highview_rgb.mp4'
    ];
    
    const fallbackVideoSrc = 'assets/video/high_views_compressed_720p/5_2_highview_rgb.mp4'; 
    const totalVideos = 13; // Double for seamless loop
    
    console.log(`Creating ${totalVideos} videos using ${videoSources.length} unique sources...`);
    
    // Create videos for seamless loop
    for (let i = 0; i < totalVideos; i++) {
        const videoContainer = document.createElement('div');
        videoContainer.className = 'marquee-video';
        
        const video = document.createElement('video');
        video.muted = true;
        video.loop = true;
        video.playsInline = true;
        video.autoplay = true;
        video.preload = 'metadata';
        
        // Cycle through available videos
        video.src = videoSources[i % videoSources.length];
        
        video.style.width = '100%';
        video.style.height = '100%';
        video.style.objectFit = 'contain'; // Changed from 'cover' to 'contain' to show full video
        
        videoContainer.appendChild(video);
        marqueeTrack.appendChild(videoContainer);
        
        // Error handling with fallback
        video.addEventListener('error', function() {
            console.warn(`Video ${i} failed to load ${this.src}, trying fallback...`);
            this.src = fallbackVideoSrc; // Switch to working video
        });
        
        // Auto-play when ready
        video.addEventListener('canplay', function() {
            this.play().catch(e => {
                console.warn(`Video ${i} autoplay failed:`, e);
            });
        });
        
        // Ensure smooth looping
        video.addEventListener('ended', function() {
            this.currentTime = 0;
            this.play();
        });
    }
    
    // Enable the scrolling animation
    const animationDuration = 60; // 60 seconds for full cycle
    marqueeTrack.style.animation = `marqueeScroll ${animationDuration}s linear infinite`;
    marqueeTrack.style.animationPlayState = 'running';
    
    console.log('✅ Video marquee initialization complete with', totalVideos, 'videos from', videoSources.length, 'unique sources');
}

// Function to update video marquee when new videos are added
function updateVideoMarquee(newVideoSources = []) {
    const marqueeTrack = document.getElementById('videoMarqueeTrack');
    if (!marqueeTrack) return;
    
    // If new video sources are provided, use them; otherwise use the placeholder
    const videoSources = newVideoSources.length > 0 ? newVideoSources : ['assets/video/high_views/highview_rgb.mp4'];
    const videoCount = Math.max(8, Math.min(14, videoSources.length * 2)); // Ensure 8-14 videos
    
    marqueeTrack.innerHTML = '';
    
    for (let i = 0; i < videoCount * 2; i++) {
        const videoContainer = document.createElement('div');
        videoContainer.className = 'marquee-video';
        
        const video = document.createElement('video');
        video.setAttribute('muted', 'true');
        video.setAttribute('autoplay', 'true');
        video.setAttribute('loop', 'true');
        video.setAttribute('playsinline', 'true');
        video.setAttribute('preload', 'metadata');
        
        const source = document.createElement('source');
        source.src = videoSources[i % videoSources.length];
        source.type = 'video/mp4';
        
        video.appendChild(source);
        videoContainer.appendChild(video);
        marqueeTrack.appendChild(videoContainer);
        
        video.addEventListener('error', function() {
            console.warn('Video failed to load:', source.src);
        });
        
        video.addEventListener('loadeddata', function() {
            this.play().catch(e => {
                console.warn('Video autoplay failed:', e);
            });
        });
    }
    
    const animationDuration = Math.max(40, videoCount * 5);
    marqueeTrack.style.animationDuration = `${animationDuration}s`;
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    initCarousel();
    initVideoMarquee();
});

// Cleanup on page unload
window.addEventListener('beforeunload', cleanup);
