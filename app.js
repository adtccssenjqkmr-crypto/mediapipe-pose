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
const uploadVideoBtn = document.getElementById("upload-video-btn");
const videoFileInput = document.getElementById("video-file-input");
const fpsCounter = document.getElementById("fps-counter");
const statusDot = document.getElementById("status-dot");
const poseDetectedVal = document.getElementById("pose-detected");

// HUD selection elements
const jointSelect = document.getElementById("joint-select");
const leftAngleVal = document.getElementById("left-angle-val");
const rightAngleVal = document.getElementById("right-angle-val");

// App State
let poseLandmarker = undefined;
let webcamRunning = false;
let isVideoMode = false;
let currentFacingMode = "user"; // "user" or "environment"
let localVideoTrack = null;
let lastVideoTime = -1;
let drawingUtils = null;
let frameCount = 0;
let lastFpsUpdate = 0;

// MediaPipe global class holders for module scope access
let FilesetResolver = null;
let PoseLandmarker = null;
let DrawingUtils = null;

// 3D Vector Operations Helper
const vec = {
    sub: (a, b) => ({ x: a.x - b.x, y: a.y - b.y, z: a.z - b.z }),
    add: (a, b) => ({ x: a.x + b.x, y: a.y + b.y, z: a.z + b.z }),
    dot: (a, b) => a.x * b.x + a.y * b.y + a.z * b.z,
    cross: (a, b) => ({
        x: a.y * b.z - a.z * b.y,
        y: a.z * b.x - a.x * b.z,
        z: a.x * b.y - a.y * b.x
    }),
    mag: (a) => Math.sqrt(a.x * a.x + a.y * a.y + a.z * a.z),
    normalize: (a) => {
        const m = Math.sqrt(a.x * a.x + a.y * a.y + a.z * a.z);
        return m === 0 ? { x: 0, y: 0, z: 0 } : { x: a.x / m, y: a.y / m, z: a.z / m };
    },
    multiplyScalar: (v, s) => ({ x: v.x * s, y: v.y * s, z: v.z * s }),
    // なす角 (3D) を計算 (度数)
    angle: (a, b) => {
        const d = vec.dot(a, b);
        const ma = vec.mag(a);
        const mb = vec.mag(b);
        if (ma === 0 || mb === 0) return 0;
        return Math.round(Math.acos(Math.min(1.0, Math.max(-1.0, d / (ma * mb)))) * (180 / Math.PI));
    }
};

// 体幹基準フレームの計算 (脊椎方向、左右軸、前方方向)
const getBodyFrame = (landmarks) => {
    const L_sh = landmarks[11];
    const R_sh = landmarks[12];
    const L_hip = landmarks[23];
    const R_hip = landmarks[24];
    
    if (!L_sh || !R_sh || !L_hip || !R_hip) return null;
    
    const shoulderCenter = {
        x: (L_sh.x + R_sh.x) / 2,
        y: (L_sh.y + R_sh.y) / 2,
        z: (L_sh.z + R_sh.z) / 2
    };
    const hipCenter = {
        x: (L_hip.x + R_hip.x) / 2,
        y: (L_hip.y + R_hip.y) / 2,
        z: (L_hip.z + R_hip.z) / 2
    };
    
    // 脊椎縦軸 (腰から肩へ向かうベクトル)
    const spine = vec.normalize(vec.sub(shoulderCenter, hipCenter));
    // 左右軸 (右から左肩へ向かうベクトル)
    const shoulderAxis = vec.normalize(vec.sub(L_sh, R_sh));
    // 体の前方向ベクトル (外積)
    const forward = vec.normalize(vec.cross(shoulderAxis, spine));
    
    return { spine, shoulderAxis, forward };
};

// 関節運動の計算
const computeJointAngles = (landmarks, mode) => {
    // 左右の関節座標の取得
    const L_sh = landmarks[11];
    const L_el = landmarks[13];
    const L_wr = landmarks[15];
    const L_hp = landmarks[23];
    const L_kn = landmarks[25];
    const L_ak = landmarks[27];
    const L_idx = landmarks[19];
    const L_ft = landmarks[31];
    
    const R_sh = landmarks[12];
    const R_el = landmarks[14];
    const R_wr = landmarks[16];
    const R_hp = landmarks[24];
    const R_kn = landmarks[26];
    const R_ak = landmarks[28];
    const R_idx = landmarks[20];
    const R_ft = landmarks[32];

    const body = getBodyFrame(landmarks);
    if (!body) return { left: null, right: null };

    // 体幹下方向
    const trunkDown = vec.normalize(vec.sub(
        { x: (L_hp.x+R_hp.x)/2, y: (L_hp.y+R_hp.y)/2, z: (L_hp.z+R_hp.z)/2 },
        { x: (L_sh.x+R_sh.x)/2, y: (L_sh.y+R_sh.y)/2, z: (L_sh.z+R_sh.z)/2 }
    ));

    const calculateSingleAngle = (isLeft) => {
        const sh = isLeft ? L_sh : R_sh;
        const el = isLeft ? L_el : R_el;
        const wr = isLeft ? L_wr : R_wr;
        const hp = isLeft ? L_hp : R_hp;
        const kn = isLeft ? L_kn : R_kn;
        const ak = isLeft ? L_ak : R_ak;
        const idx = isLeft ? L_idx : R_idx;
        const ft = isLeft ? L_ft : R_ft;

        // すべての参照ランドマークが存在することを確認するガード条件（手先・足先も見切れ対策で含める）
        if (!sh || !el || !wr || !hp || !kn || !ak || !idx || !ft) return null;

        switch (mode) {
            case "elbow_flexion": {
                // 肘関節 屈曲: 肩 - 肘 - 手首
                const arm = vec.sub(sh, el);
                const forearm = vec.sub(wr, el);
                return vec.angle(arm, forearm);
            }
            case "knee_flexion": {
                // 膝関節 屈曲: 股関節 - 膝 - 足首
                const thigh = vec.sub(hp, kn);
                const calf = vec.sub(ak, kn);
                return vec.angle(thigh, calf);
            }
            case "wrist_flexion": {
                // 手関節 屈曲: 肘 - 手首 - 人差し指
                const forearm = vec.sub(el, wr);
                const hand = vec.sub(idx, wr);
                return vec.angle(forearm, hand);
            }
            case "ankle_flexion": {
                // 足関節 屈曲 (底背屈): 膝 - 足首 - つま先
                const calf = vec.sub(kn, ak);
                const foot = vec.sub(ft, ak);
                return vec.angle(calf, foot);
            }
            case "shoulder_flexion": {
                // 肩関節 屈曲/伸展: 上腕の前後スイング角度 (矢状面射影)
                const armVec = vec.sub(el, sh);
                const planeAxis = body.shoulderAxis;
                const armProj = vec.normalize(vec.sub(armVec, vec.multiplyScalar(planeAxis, vec.dot(armVec, planeAxis))));
                const angle = vec.angle(trunkDown, armProj);
                const isFlexion = vec.dot(armProj, body.forward) > 0;
                return isFlexion ? `屈曲 ${angle}` : `伸展 ${angle}`;
            }
            case "shoulder_abduction": {
                // 肩関節 外転/内転: 上腕の左右開き角度 (冠状面射影)
                const armVec = vec.sub(el, sh);
                const planeAxis = body.forward;
                const armProj = vec.normalize(vec.sub(armVec, vec.multiplyScalar(planeAxis, vec.dot(armVec, planeAxis))));
                const angle = vec.angle(trunkDown, armProj);
                const isAbduction = isLeft ? (vec.dot(armProj, body.shoulderAxis) > 0) : (vec.dot(armProj, body.shoulderAxis) < 0);
                return isAbduction ? `外転 ${angle}` : `内転 ${angle}`;
            }
            case "shoulder_rotation": {
                // 肩関節 内旋/外旋: 上腕軸を法線とする平面への前腕の射影
                const arm = vec.normalize(vec.sub(el, sh));
                const forearm = vec.normalize(vec.sub(wr, el));
                const forearmProj = vec.normalize(vec.sub(forearm, vec.multiplyScalar(arm, vec.dot(forearm, arm))));
                const forwardProj = vec.normalize(vec.sub(body.forward, vec.multiplyScalar(arm, vec.dot(body.forward, arm))));
                const angle = vec.angle(forwardProj, forearmProj);
                
                const cross = vec.cross(forwardProj, forearmProj);
                const dotSign = vec.dot(cross, arm);
                const isInternal = isLeft ? (dotSign > 0) : (dotSign < 0);
                return isInternal ? `内旋 ${angle}` : `外旋 ${angle}`;
            }
            case "hip_flexion": {
                // 股関節 屈曲/伸展: 大腿の前後角度 (矢状面射影)
                const thighVec = vec.sub(kn, hp);
                const planeAxis = body.shoulderAxis;
                const thighProj = vec.normalize(vec.sub(thighVec, vec.multiplyScalar(planeAxis, vec.dot(thighVec, planeAxis))));
                const angle = vec.angle(trunkDown, thighProj);
                const isFlexion = vec.dot(thighProj, body.forward) > 0;
                return isFlexion ? `屈曲 ${angle}` : `伸展 ${angle}`;
            }
            case "hip_abduction": {
                // 股関節 外転/内転: 大腿の左右角度 (冠状面射影)
                const thighVec = vec.sub(kn, hp);
                const planeAxis = body.forward;
                const thighProj = vec.normalize(vec.sub(thighVec, vec.multiplyScalar(planeAxis, vec.dot(thighVec, planeAxis))));
                const angle = vec.angle(trunkDown, thighProj);
                const isAbduction = isLeft ? (vec.dot(thighProj, body.shoulderAxis) > 0) : (vec.dot(thighProj, body.shoulderAxis) < 0);
                return isAbduction ? `外転 ${angle}` : `内転 ${angle}`;
            }
            case "hip_rotation": {
                // 股関節 内旋/外旋: 大腿軸を法線とする平面への下腿の射影
                const thigh = vec.normalize(vec.sub(kn, hp));
                const calf = vec.normalize(vec.sub(ak, kn));
                const calfProj = vec.normalize(vec.sub(calf, vec.multiplyScalar(thigh, vec.dot(calf, thigh))));
                const forwardProj = vec.normalize(vec.sub(body.forward, vec.multiplyScalar(thigh, vec.dot(body.forward, thigh))));
                const angle = vec.angle(forwardProj, calfProj);
                
                const cross = vec.cross(forwardProj, calfProj);
                const dotSign = vec.dot(cross, thigh);
                const isInternal = isLeft ? (dotSign > 0) : (dotSign < 0);
                return isInternal ? `内旋 ${angle}` : `外旋 ${angle}`;
            }
            default:
                return null;
        }
    };

    return {
        left: calculateSingleAngle(true),
        right: calculateSingleAngle(false)
    };
};

// HUD表示の更新
const updateHudAngles = (landmarks) => {
    const activeMode = jointSelect.value;
    const angles = computeJointAngles(landmarks, activeMode);

    const formatValue = (val) => {
        if (val === null) return "--°";
        if (typeof val === "string") {
            const [direction, num] = val.split(" ");
            return `${direction}\n${num}°`;
        }
        return `${val}°`;
    };

    leftAngleVal.innerText = formatValue(angles.left);
    rightAngleVal.innerText = formatValue(angles.right);
};

// ミラーリング（反転）状態の自動更新
const updateMirrorMode = () => {
    const videoWrapper = video.parentElement;
    if (webcamRunning && currentFacingMode === "user" && !isVideoMode) {
        videoWrapper.classList.add("mirror");
    } else {
        videoWrapper.classList.remove("mirror");
    }
};

// リアルタイム骨格検出ループ (クラッシュ防止＆フレームレート最適化版)
const predictWebcam = async () => {
    if (!webcamRunning) return;

    // クラッシュ防止ガード: ビデオデータがまだ読み込まれていない、またはアスペクト比が0x0の場合は次フレームをスケジュールして即復帰
    if (video.readyState < 2 || video.videoWidth === 0 || video.videoHeight === 0) {
        if (webcamRunning) {
            window.requestAnimationFrame(predictWebcam);
        }
        return;
    }

    // キャンバス解像度の同期
    if (canvasElement.width !== video.videoWidth || canvasElement.height !== video.videoHeight) {
        canvasElement.width = video.videoWidth;
        canvasElement.height = video.videoHeight;
    }

    const startTimeMs = performance.now();

    // 映像データが更新されたフレームのみ姿勢推定を実行 (不要な重複計算のカット)
    if (video.currentTime !== lastVideoTime) {
        lastVideoTime = video.currentTime;
        
        try {
            // 姿勢の検出
            const results = poseLandmarker.detectForVideo(video, startTimeMs);
            
            // 描画
            canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

            if (results.landmarks && results.landmarks.length > 0) {
                poseDetectedVal.innerText = isVideoMode ? "分析中" : "検出中";
                poseDetectedVal.classList.add("active");
                
                const landmarks = results.landmarks[0];
                
                // 骨格接続線の描画
                drawingUtils.drawConnectors(
                    landmarks, 
                    PoseLandmarker.POSE_CONNECTIONS, 
                    { color: "#9d00ff", lineWidth: 4 }
                );
                drawingUtils.drawLandmarks(
                    landmarks, 
                    { color: "#00f0ff", lineWidth: 2, radius: 4 }
                );

                // 各種関節角度の計算とHUD表示の更新
                updateHudAngles(landmarks);
            } else {
                poseDetectedVal.innerText = "未検出";
                poseDetectedVal.classList.remove("active");
                leftAngleVal.innerText = "--°";
                rightAngleVal.innerText = "--°";
            }
        } catch (detectError) {
            console.warn("Detection error (ignored during transition):", detectError);
        }

        // FPS計算
        calculateFps(startTimeMs);
    }

    // 次のフレーム要求
    if (webcamRunning) {
        window.requestAnimationFrame(predictWebcam);
    }
};

// カメラ制御: On/Off切り替え
const toggleWebcam = async () => {
    if (!poseLandmarker) return;

    if (webcamRunning && !isVideoMode) {
        // カメラ停止
        webcamRunning = false;
        toggleBtn.innerHTML = `
            <svg class="btn-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <polygon points="5 3 19 12 5 21 5 3"></polygon>
            </svg>
            <span>カメラ開始</span>
        `;
        toggleBtn.className = "panel-btn primary-btn";
        statusDot.classList.remove("active");
        
        stopMediaStream();
        canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
        fpsCounter.innerText = "-- FPS";
        poseDetectedVal.innerText = "未検出";
        switchCameraBtn.disabled = true;
    } else {
        // カメラ開始
        webcamRunning = true;
        isVideoMode = false;
        video.controls = false;
        
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
    updateMirrorMode();
};

// メディアストリームの完全停止
const stopMediaStream = () => {
    if (localVideoTrack) {
        localVideoTrack.stop();
        localVideoTrack = null;
    }
    video.srcObject = null;
    video.src = "";
};

// カメラ起動
const startCamera = async () => {
    stopMediaStream();

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
        video.addEventListener("loadeddata", predictWebcam);
    } catch (err) {
        console.error("Camera access error:", err);
        alert("カメラへのアクセスを許可してください。");
        toggleWebcam();
    }
};

// カメラ切り替え (フロント/リア)
const switchCamera = async () => {
    if (!webcamRunning || isVideoMode) return;
    currentFacingMode = currentFacingMode === "user" ? "environment" : "user";
    updateMirrorMode();
    await startCamera();
};

// 動画ファイルのインポートと再生開始
const handleVideoUpload = (event) => {
    const file = event.target.files[0];
    if (!file) return;

    // ストリームの停止
    stopMediaStream();
    webcamRunning = true;
    isVideoMode = true;
    updateMirrorMode();

    // 動画モードに切り替え
    video.src = URL.createObjectURL(file);
    video.controls = true; // スマホでの操作性を高めるためにコントローラーを有効化
    
    // UIの切り替え
    statusDot.classList.add("active");
    toggleBtn.innerHTML = `
        <svg class="btn-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <polygon points="5 3 19 12 5 21 5 3"></polygon>
        </svg>
        <span>カメラ開始</span>
    `;
    toggleBtn.className = "panel-btn secondary-btn";
    switchCameraBtn.disabled = true;

    video.addEventListener("loadeddata", () => {
        video.play();
        predictWebcam();
    });
};

// FPS計算
const calculateFps = (now) => {
    frameCount++;
    if (now - lastFpsUpdate >= 1000) {
        const fps = Math.round((frameCount * 1000) / (now - lastFpsUpdate));
        fpsCounter.innerText = `${fps} FPS`;
        frameCount = 0;
        lastFpsUpdate = now;
    }
};

// Initialize Pose Landmarker
const initPoseLandmarker = async () => {
    try {
        loadingStatus.innerText = "MediaPipeライブラリをロード中...";
        const mediaPipe = await import("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.8/vision_bundle.mjs");
        FilesetResolver = mediaPipe.FilesetResolver;
        PoseLandmarker = mediaPipe.PoseLandmarker;
        DrawingUtils = mediaPipe.DrawingUtils;

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

// Event Listeners
toggleBtn.addEventListener("click", toggleWebcam);
switchCameraBtn.addEventListener("click", switchCamera);
uploadVideoBtn.addEventListener("click", () => videoFileInput.click());
videoFileInput.addEventListener("change", handleVideoUpload);

// 関節切り替え時にも角度表示を即時クリア/再計算させる
jointSelect.addEventListener("change", () => {
    leftAngleVal.innerText = "--°";
    rightAngleVal.innerText = "--°";
});

// Disable Start button initially until model is loaded
toggleBtn.disabled = true;

// Start initialization
initPoseLandmarker();
