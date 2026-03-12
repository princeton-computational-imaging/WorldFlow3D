// Optional performance monitoring script
// Uncomment the line below in index.html to enable performance monitoring

let lastFrameTime = performance.now();
let frameCount = 0;
let fpsDisplay = null;

// Create FPS counter display
function createFpsDisplay() {
    fpsDisplay = document.createElement('div');
    fpsDisplay.id = 'fps-counter';
    fpsDisplay.style.cssText = `
        position: fixed;
        top: 10px;
        right: 10px;
        background: rgba(0, 0, 0, 0.8);
        color: white;
        padding: 8px 12px;
        font-family: monospace;
        font-size: 12px;
        border-radius: 4px;
        z-index: 1000;
        pointer-events: none;
    `;
    document.body.appendChild(fpsDisplay);
}

// Update performance metrics
function updatePerformanceMetrics() {
    const currentTime = performance.now();
    frameCount++;
    
    if (currentTime - lastFrameTime >= 1000) {
        const fps = Math.round(frameCount * 1000 / (currentTime - lastFrameTime));
        const memory = performance.memory ? 
            Math.round(performance.memory.usedJSHeapSize / 1048576) : 'N/A';
        
        if (fpsDisplay) {
            fpsDisplay.innerHTML = `
                FPS: ${fps}<br>
                Memory: ${memory}MB
            `;
        }
        
        frameCount = 0;
        lastFrameTime = currentTime;
        
        // Log to console every 10 seconds
        if (Math.floor(currentTime / 10000) !== Math.floor((currentTime - 1000) / 10000)) {
            console.log(`Performance: ${fps} FPS, ${memory}MB memory`);
        }
    }
    
    requestAnimationFrame(updatePerformanceMetrics);
}

// Initialize performance monitoring
document.addEventListener('DOMContentLoaded', () => {
    createFpsDisplay();
    requestAnimationFrame(updatePerformanceMetrics);
    
    console.log('Performance monitoring enabled. Check top-right corner for FPS counter.');
    console.log('Performance metrics will be logged to console every 10 seconds.');
});

export { updatePerformanceMetrics };
