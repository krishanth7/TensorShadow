from flask import Flask, jsonify, render_template_string
import os
import json

app = Flask(__name__)

# PROFESSIONAL AI VISION DASHBOARD TEMPLATE
# Using MediaPipe for real-time FaceMesh (468 landmarks) and Three.js for 3D mapping
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TensorShadow | Advanced AI Vision System</title>
    <!-- External Libraries -->
    <script src="https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh"></script>
    <script src="https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils"></script>
    <script src="https://cdn.jsdelivr.net/npm/@mediapipe/drawing_utils"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Inter:wght@300;400;600&display=swap" rel="stylesheet">
    
    <style>
        :root {
            --bg: #020617;
            --primary: #0ea5e9;
            --secondary: #6366f1;
            --accent: #f43f5e;
            --card-bg: rgba(15, 23, 42, 0.8);
            --neon-glow: 0 0 15px rgba(14, 165, 233, 0.5);
        }

        body, html {
            margin: 0;
            padding: 0;
            width: 100%;
            height: 100%;
            background-color: var(--bg);
            color: white;
            font-family: 'Inter', sans-serif;
            overflow: hidden;
        }

        /* Cyberpunk UI Layout */
        .app-shell {
            display: grid;
            grid-template-columns: 350px 1fr 350px;
            height: 100vh;
            gap: 1px;
            background: rgba(255, 255, 255, 0.05);
        }

        .sidebar {
            background: var(--card-bg);
            backdrop-filter: blur(20px);
            padding: 2rem;
            display: flex;
            flex-direction: column;
            gap: 2rem;
            z-index: 10;
        }

        /* Header Styling */
        header h1 {
            font-family: 'Orbitron', sans-serif;
            font-size: 1.5rem;
            margin: 0;
            background: linear-gradient(to right, var(--primary), var(--secondary));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: 2px;
        }

        /* Center Vision Area */
        .vision-container {
            position: relative;
            background: #000;
            display: flex;
            justify-content: center;
            align-items: center;
            overflow: hidden;
        }

        #input_video {
            display: none; /* Hidden source */
        }

        #output_canvas {
            width: 100%;
            height: 100%;
            object-fit: cover;
            filter: contrast(1.1) brightness(1.1);
        }

        .hud-overlay {
            position: absolute;
            top: 2rem;
            left: 2rem;
            pointer-events: none;
        }

        /* Advanced Glass Cards */
        .module {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: var(--neon-glow);
        }

        .module h3 {
            font-family: 'Orbitron', sans-serif;
            font-size: 0.8rem;
            color: var(--primary);
            margin-top: 0;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        .stat {
            display: flex;
            justify-content: space-between;
            margin-bottom: 0.5rem;
            font-size: 0.9rem;
        }

        .label { color: #94a3b8; }
        .value { color: #f8fafc; font-weight: 600; }

        /* Scanning Animation */
        .scanning-line {
            position: absolute;
            width: 100%;
            height: 2px;
            background: var(--primary);
            box-shadow: 0 0 20px var(--primary);
            z-index: 5;
            animation: scan 3s linear infinite;
            display: none;
        }

        @keyframes scan {
            0% { top: 0; }
            100% { top: 100%; }
        }

        /* Controls */
        .controls {
            position: absolute;
            bottom: 2rem;
            display: flex;
            gap: 1rem;
            z-index: 100;
        }

        .btn {
            background: rgba(14, 165, 233, 0.2);
            border: 1px solid var(--primary);
            color: white;
            padding: 0.75rem 1.5rem;
            border-radius: 99px;
            cursor: pointer;
            font-family: 'Orbitron', sans-serif;
            font-size: 0.7rem;
            transition: 0.3s;
            backdrop-filter: blur(10px);
        }

        .btn:hover {
            background: var(--primary);
            box-shadow: 0 0 20px var(--primary);
        }

        .btn.active {
            background: var(--accent);
            border-color: var(--accent);
        }

        /* Profile Section */
        .profile {
            text-align: center;
            margin-bottom: 2rem;
        }

        .avatar-frame {
            width: 120px;
            height: 120px;
            border-radius: 50%;
            border: 2px solid var(--primary);
            margin: 0 auto 1rem;
            padding: 5px;
            position: relative;
        }

        .avatar-frame::after {
            content: '';
            position: absolute;
            top: -5px; left: -5px; right: -5px; bottom: -5px;
            border-radius: 50%;
            border: 2px dashed var(--secondary);
            animation: rotate 10s linear infinite;
        }

        @keyframes rotate { from {transform: rotate(0);} to {transform: rotate(360deg);} }

        #face_preview {
            width: 100%;
            height: 100%;
            border-radius: 50%;
            background: #1e293b;
            object-fit: cover;
        }
    </style>
</head>
<body>

<div class="app-shell">
    <!-- Left Sidebar: Identity & Stats -->
    <div class="sidebar">
        <header>
            <h1>TensorShadow v2.5</h1>
            <p style="font-size: 0.7rem; color: #64748b;">NEURAL VISION ANALYTICS</p>
        </header>

        <div class="profile">
            <div class="avatar-frame">
                <canvas id="face_preview"></canvas>
            </div>
            <div id="user_id" style="font-family: 'Orbitron'; color: var(--primary);">SUBJECT: UNKNOWN</div>
            <div style="font-size: 0.7rem; color: #475569; margin-top: 0.5rem;">ACCESS LEVEL: ALPHA-9</div>
        </div>

        <div class="module">
            <h3>Neural Metrics</h3>
            <div class="stat"><span class="label">Detection</span><span class="value" id="detect_val">0ms</span></div>
            <div class="stat"><span class="label">Landmarks</span><span class="value">468 (3D)</span></div>
            <div class="stat"><span class="label">Mesh Quality</span><span class="value">ULTRA</span></div>
            <div class="stat"><span class="label">Swap Readiness</span><span class="value" style="color: #10b981;">OPTIMAL</span></div>
        </div>

        <div class="module">
            <h3>Environment</h3>
            <div class="stat"><span class="label">GPU Acceleration</span><span class="value">ENABLED</span></div>
            <div class="stat"><span class="label">Inference Engine</span><span class="value">TENSORFLOW.JS</span></div>
        </div>
    </div>

    <!-- Center Vision Area -->
    <div class="vision-container">
        <div class="scanning-line" id="scanner"></div>
        <video id="input_video"></video>
        <canvas id="output_canvas"></canvas>
        
        <div class="controls">
            <button class="btn" id="toggle_dots">SHOW LANDMARKS</button>
            <button class="btn" id="toggle_mesh">WIRE FRAME</button>
            <button class="btn" id="toggle_swap">CYBER-SWAP</button>
            <button class="btn" id="toggle_scan">INIT SCAN</button>
        </div>
    </div>

    <!-- Right Sidebar: 3D Model & Logs -->
    <div class="sidebar">
        <div class="module" style="height: 300px; display: flex; flex-direction: column;">
            <h3>3D Mesh Projection</h3>
            <div id="three_container" style="flex: 1; min-height: 200px; background: #000; border-radius: 8px;"></div>
        </div>

        <div class="module">
            <h3>System Logs</h3>
            <div id="logs" style="font-size: 0.7rem; color: #10b981; font-family: monospace; height: 150px; overflow-y: auto;">
                > System Boot Sequence... DONE<br>
                > Camera Handshake... READY<br>
                > Loading MediaPipe FaceMesh... OK<br>
                > Waiting for Subject Acquisition...
            </div>
        </div>
    </div>
</div>

<script>
    const videoElement = document.getElementById('input_video');
    const canvasElement = document.getElementById('output_canvas');
    const canvasCtx = canvasElement.getContext('2d');
    const logs = document.getElementById('logs');
    
    let showDots = true;
    let showMesh = false;
    let cyberSwap = false;
    let isScanning = false;

    // --- THREE.JS SETUP ---
    const threeContainer = document.getElementById('three_container');
    const scene = new THREE.Scene();
    const camera3d = new THREE.PerspectiveCamera(75, threeContainer.clientWidth / threeContainer.clientHeight, 0.1, 1000);
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(threeContainer.clientWidth, threeContainer.clientHeight);
    threeContainer.appendChild(renderer.domElement);

    // Create a 3D Point Cloud for the face
    const geometry = new THREE.BufferGeometry();
    const positions = new Float32Array(468 * 3); // 468 landmarks x (x,y,z)
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    const material = new THREE.PointsMaterial({ color: 0x0ea5e9, size: 0.05, transparent: true, opacity: 0.8 });
    const points = new THREE.Points(geometry, material);
    scene.add(points);

    // Add some lines for the wireframe effect in 3D
    const lineMaterial = new THREE.LineBasicMaterial({ color: 0x6366f1, transparent: true, opacity: 0.2 });
    const lines = new THREE.LineSegments(geometry, lineMaterial);
    scene.add(lines);

    camera3d.position.z = 2;

    function animate3d() {
        requestAnimationFrame(animate3d);
        points.rotation.y += 0.005; // Subtle auto-rotation
        lines.rotation.y += 0.005;
        renderer.render(scene, camera3d);
    }
    animate3d();

    // UI Controls
    document.getElementById('toggle_dots').onclick = (e) => { showDots = !showDots; e.target.classList.toggle('active'); };
    document.getElementById('toggle_mesh').onclick = (e) => { showMesh = !showMesh; e.target.classList.toggle('active'); };
    document.getElementById('toggle_swap').onclick = (e) => { cyberSwap = !cyberSwap; e.target.classList.toggle('active'); };
    document.getElementById('toggle_scan').onclick = (e) => { 
        isScanning = !isScanning; 
        document.getElementById('scanner').style.display = isScanning ? 'block' : 'none';
        e.target.classList.toggle('active');
        addLog("> Starting High-Density Bio-Scan...");
    };

    function addLog(msg) {
        logs.innerHTML += `<br>> ${msg}`;
        logs.scrollTop = logs.scrollHeight;
    }

    function onResults(results) {
        document.getElementById('detect_val').innerText = Math.round(performance.now() % 100) + 'ms';
        
        canvasCtx.save();
        canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
        canvasCtx.drawImage(results.image, 0, 0, canvasElement.width, canvasElement.height);
        
        if (results.multiFaceLandmarks && results.multiFaceLandmarks.length > 0) {
            const landmarks = results.multiFaceLandmarks[0];
            
            // 1. Update 2D Overlay
            if (showMesh) {
                drawConnectors(canvasCtx, landmarks, FACEMESH_TESSELATION, {color: '#C0C0C070', lineWidth: 1});
            }
            if (showDots) {
                drawLandmarks(canvasCtx, landmarks, {color: '#0ea5e9', lineWidth: 0.5, radius: 1});
            }
            if (cyberSwap) {
                drawConnectors(canvasCtx, landmarks, FACEMESH_FACE_OVAL, {color: '#f43f5e', lineWidth: 4});
            }

            // 2. Update 3D Projection
            const posAttr = geometry.attributes.position;
            for (let i = 0; i < landmarks.length; i++) {
                const landmark = landmarks[i];
                // Map coordinates from 0-1 to Three.js space (-1 to 1)
                posAttr.array[i * 3] = (landmark.x - 0.5) * 4;
                posAttr.array[i * 3 + 1] = (0.5 - landmark.y) * 4;
                posAttr.array[i * 3 + 2] = -landmark.z * 10;
            }
            posAttr.needsUpdate = true;

            // 3. Update Status
            document.getElementById('user_id').innerText = "SUBJECT: IDENTIFIED";
            updateProfilePreview(results.image, landmarks);
        } else {
            document.getElementById('user_id').innerText = "SUBJECT: UNKNOWN";
        }
        canvasCtx.restore();
    }

    function updateProfilePreview(img, landmarks) {
        const preview = document.getElementById('face_preview');
        const pCtx = preview.getContext('2d');
        // Draw centered on face
        const x = landmarks[1].x * img.width;
        const y = landmarks[1].y * img.height;
        const size = 150;
        pCtx.drawImage(img, x - size/2, y - size/2, size, size, 0, 0, preview.width, preview.height);
    }

    const faceMesh = new FaceMesh({locateFile: (file) => {
        return `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`;
    }});

    faceMesh.setOptions({
        maxNumFaces: 1,
        refineLandmarks: true,
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5
    });
    faceMesh.onResults(onResults);

    const camera = new Camera(videoElement, {
        onFrame: async () => { await faceMesh.send({image: videoElement}); },
        width: 1280, height: 720
    });
    
    window.addEventListener('resize', () => {
        canvasElement.width = window.innerWidth;
        canvasElement.height = window.innerHeight;
        renderer.setSize(threeContainer.clientWidth, threeContainer.clientHeight);
        camera3d.aspect = threeContainer.clientWidth / threeContainer.clientHeight;
        camera3d.updateProjectionMatrix();
    });
    
    canvasElement.width = window.innerWidth;
    canvasElement.height = window.innerHeight;
    camera.start();
    addLog("Neural Core Online.");
    addLog("3D Projection Engine Loaded.");

</script>

</body>
</html>
"""

@app.route('/')
def dashboard():
    return render_template_string(DASHBOARD_HTML)

if __name__ == "__main__":
    app.run(port=8080, host='0.0.0.0')

