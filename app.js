// MediaPipe classes will be loaded dynamically inside initPoseLandmarker

// Global error handlers for debugging on mobile
window.addEventListener("error", (event) => {
    const status = document.getElementById("loading-status");
    if (status) {
        status.style.color = "#ff4a4a";
        status.innerText = `ランタイムエラー: ${event.message}\nファイル: ${event.filename}:${event.lineno}`;
    }
    alert("エラーが発生しました:\n" + event.message);
});

window.addEventListener("unhandledrejection", (event) => {
    const status = document.getElementById("loading-status");
    if (status) {
        status.style.color = "#ff4a4a";
        status.innerText = `非同期エラー: ${event.reason}`;
    }
    alert("非同期エラーが発生しました:\n" + event.reason);
});

// DOM Elements
const video = document.getElementById("webcam");
const canvasElement = document.getElementById("output_canvas");
const canvasCtx = canvasElement.getContext("2d");
const loadingScreen = document.getElementById("loading-screen");
const loadingStatus = document.getElementById("loading-status");
const toggleBtn = document.getElementById("toggle-btn");
const switchCameraBtn = document.getElementById("switch-camera-btn");
const fpsCounter = document.getElementById("fps-counter");
const statusDot = document.getElementById("status-dot");
const poseDetectedVal = document.getElementById("pose-detected");

// HUD Angle Elements
const leftElbowAngleVal = document.getElementById("left-elbow-angle");
const rightElbowAngleVal = document.getElementById("right-elbow-angle");
const leftKneeAngleVal = document.getElementById("left-knee-angle");
const rightKneeAngleVal = document.getElementById("right-knee-angle");

// App State
let poseLandmarker = undefined;
let webcamRunning = false;
let currentFacingMode = "user"; // "user" or "environment"
let localVideoTrack = null;
let lastVideoTime = -1;
let drawingUtils = null;
let frameCount = 0;
let lastFpsUpdate = 0;

// Initialize Pose Landmarker
const initPoseLandmarker = async () => {
    try {
        loadingStatus.innerText = "MediaPipeライブラリをロード中...";
        const mediaPipe = await import("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.8/vision_bundle.mjs");
        const { PoseLandmarker, FilesetResolver, DrawingUtils } = mediaPipe;

        loadingStatus.innerText = "WebAssemblyリソースを取得中...";
        const vision = await FilesetResolver.forVisionTasks(
            "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.8/wasm"
        );
        
        loadingStatus.innerText = "AI姿勢推定モデルをロード中...";
        poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
            baseOptions: {
                modelAssetPath: `https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task`,
                delegate: "GPU"
            },
            runningMode: "VIDEO",
            numPoses: 1
        });
        
        drawingUtils = new DrawingUtils(canvasCtx);
        
        // Setup successful, remove loader
        loadingScreen.classList.add("inactive");
        console.log("MediaPipe Pose Landmarker successfully initialized.");
        
        // Enable camera controls
        toggleBtn.disabled = false;
    } catch (error) {
        console.error("Initialization error:", error);
        loadingStatus.style.color = "#ff4a4a";
        loadingStatus.innerText = `初期化エラー: ${error.message}\n\nモデルURLやCDNへのアクセスがブロックされている可能性があります。通信状態を確認してください。`;
        alert("初期化エラーが発生しました:\n" + error.message);
    }
};

// Start initialization
initPoseLandmarker();

// Helper to calculate 2D angle between three joints
const calculateAngle = (p1, p2, p3) => {
    if (!p1 || !p2 || !p3) return null;
    
    // Vector BA (p1 to p2) and BC (p3 to p2)
    const ba = { x: p1.x - p2.x, y: p1.y - p2.y };
    const bc = { x: p3.x - p2.x, y: p3.y - p2.y };
    
    // Dot product and Magnitudes
    const dotProduct = ba.x * bc.x + ba.y * bc.y;
    const magBA = Math.sqrt(ba.x * ba.x + ba.y * ba.y);
    const magBC = Math.sqrt(bc.x * bc.x + bc.y * bc.y);
    
    if (magBA === 0 || magBC === 0) return null;
    
    const cosine = dotProduct / (magBA * magBC);
    // Clamp to avoid float precision issues
    const clampedCosine = Math.min(1.0, Math.max(-1.0, cosine));
    
    const angleRad = Math.acos(clampedCosine);
    const angleDeg = Math.round(angleRad * (180 / Math.PI));
    
    return angleDeg;
};

// Update UI with calculated joint angles
const updateAngles = (landmarks) => {
    // Indices reference: 11:L_Shoulder, 13:L_Elbow, 15:L_Wrist
    // 12:R_Shoulder, 14:R_Elbow, 16:R_Wrist
    // 23:L_Hip, 25:L_Knee, 27:L_Ankle
    // 24:R_Hip, 26:R_Knee, 28:R_Ankle
    const L_shoulder = landmarks[11];
    const L_elbow = landmarks[13];
    const L_wrist = landmarks[15];
    const R_shoulder = landmarks[12];
    const R_elbow = landmarks[14];
    const R_wrist = landmarks[16];
    
    const L_hip = landmarks[23];
    const L_knee = landmarks[25];
    const L_ankle = landmarks[27];
    const R_hip = landmarks[24];
    const R_knee = landmarks[26];
    const R_ankle = landmarks[28];

    // Calculate angles
    const leftElbowAngle = calculateAngle(L_shoulder, L_elbow, L_wrist);
    const rightElbowAngle = calculateAngle(R_shoulder, R_elbow, R_wrist);
    const leftKneeAngle = calculateAngle(L_hip, L_knee, L_ankle);
    const rightKneeAngle = calculateAngle(R_hip, R_knee, R_ankle);

    // Update UI elements
    leftElbowAngleVal.innerText = leftElbowAngle !== null ? `${leftElbowAngle}°` : "--°";
    rightElbowAngleVal.innerText = rightElbowAngle !== null ? `${rightElbowAngle}°` : "--°";
    leftKneeAngleVal.innerText = leftKneeAngle !== null ? `${leftKneeAngle}°` : "--°";
    rightKneeAngleVal.innerText = rightKneeAngle !== null ? `${rightKneeAngle}°` : "--°";
};

// Check if webcam is available and start/stop
const toggleWebcam = async () => {
    if (!poseLandmarker) return;

    if (webcamRunning) {
        // Stop Camera
        webcamRunning = false;
        toggleBtn.innerHTML = `
            <svg class="btn-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <polygon points="5 3 19 12 5 21 5 3"></polygon>
            </svg>
            <span>カメラ開始</span>
        `;
        toggleBtn.className = "panel-btn primary-btn";
        statusDot.classList.remove("active");
        
        if (localVideoTrack) {
            localVideoTrack.stop();
            localVideoTrack = null;
        }
        video.srcObject = null;
        canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
        fpsCounter.innerText = "-- FPS";
        poseDetectedVal.innerText = "未検出";
        switchCameraBtn.disabled = true;
    } else {
        // Start Camera
        webcamRunning = true;
        toggleBtn.innerHTML = `
            <svg class="btn-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <rect x="4" y="4" width="16" height="16" rx="2" ry="2"></rect>
            </svg>
            <span>一時停止</span>
        `;
        toggleBtn.className = "panel-btn secondary-btn";
        statusDot.classList.add("active");
        
        await startCamera();
        switchCameraBtn.disabled = false;
    }
};

// Start camera stream
const startCamera = async () => {
    if (localVideoTrack) {
        localVideoTrack.stop();
    }

    const constraints = {
        video: {
            facingMode: currentFacingMode,
            width: { ideal: 640 },
            height: { ideal: 480 }
        }
    };

    try {
        const stream = await navigator.mediaDevices.getUserMedia(constraints);
        video.srcObject = stream;
        localVideoTrack = stream.getVideoTracks()[0];
        
        // Mirror front-facing camera
        if (currentFacingMode === "user") {
            video.classList.remove("rear-camera");
        } else {
            video.classList.add("rear-camera");
        }

        video.addEventListener("loadeddata", predictWebcam);
    } catch (err) {
        console.error("Camera access error:", err);
        alert("カメラへのアクセスを許可してください。");
        toggleWebcam();
    }
};

// Switch Front/Rear Camera
const switchCamera = async () => {
    if (!webcamRunning) return;
    currentFacingMode = currentFacingMode === "user" ? "environment" : "user";
    await startCamera();
};

// Main loop for real-time predictions
const predictWebcam = async () => {
    if (!webcamRunning) return;

    // Adjust canvas dimensions to match video dimensions
    if (canvasElement.width !== video.videoWidth || canvasElement.height !== video.videoHeight) {
        canvasElement.width = video.videoWidth;
        canvasElement.height = video.videoHeight;
    }

    const startTimeMs = performance.now();

    if (lastVideoTime !== video.currentTime) {
        lastVideoTime = video.currentTime;
        
        // Detect Pose
        const results = poseLandmarker.detectForVideo(video, startTimeMs);
        
        // Clear canvas
        canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

        if (results.landmarks && results.landmarks.length > 0) {
            poseDetectedVal.innerText = "検出中";
            poseDetectedVal.style.color = "var(--text-active)";
            
            const landmarks = results.landmarks[0];
            
            // Draw skeleton landmarks and connections
            drawingUtils.drawConnectors(
                landmarks, 
                PoseLandmarker.POSE_CONNECTIONS, 
                { color: "#9d00ff", lineWidth: 4 }
            );
            drawingUtils.drawLandmarks(
                landmarks, 
                { color: "#00f0ff", lineWidth: 2, radius: 4 }
            );

            // Compute angles and update HUD
            updateAngles(landmarks);
        } else {
            poseDetectedVal.innerText = "未検出";
            poseDetectedVal.style.color = "var(--text-muted)";
            leftElbowAngleVal.innerText = "--°";
            rightElbowAngleVal.innerText = "--°";
            leftKneeAngleVal.innerText = "--°";
            rightKneeAngleVal.innerText = "--°";
        }

        // Calculate and display FPS
        calculateFps(startTimeMs);
    }

    // Call next frame
    if (webcamRunning) {
        window.requestAnimationFrame(predictWebcam);
    }
};

// Calculate and update FPS HUD
const calculateFps = (now) => {
    frameCount++;
    if (now - lastFpsUpdate >= 1000) {
        const fps = Math.round((frameCount * 1000) / (now - lastFpsUpdate));
        fpsCounter.innerText = `${fps} FPS`;
        frameCount = 0;
        lastFpsUpdate = now;
    }
};

// Event Listeners
toggleBtn.addEventListener("click", toggleWebcam);
switchCameraBtn.addEventListener("click", switchCamera);

// Disable Start button initially until model is loaded
toggleBtn.disabled = true;
