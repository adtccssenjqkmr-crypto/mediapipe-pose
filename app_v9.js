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
const recordBtn = document.getElementById("record-btn");
const uploadVideoBtn = document.getElementById("upload-video-btn");
const uploadImageBtn = document.getElementById("upload-image-btn");
const videoFileInput = document.getElementById("video-file-input");
const imageFileInput = document.getElementById("image-file-input");
const fpsCounter = document.getElementById("fps-counter");
const statusDot = document.getElementById("status-dot");
const poseDetectedVal = document.getElementById("pose-detected");

// HUD selection elements
const jointSelect = document.getElementById("joint-select");
const leftAngleVal = document.getElementById("left-angle-val");
const rightAngleVal = document.getElementById("right-angle-val");

// Chart Drawer Elements
const chartSection = document.getElementById("chart-section");
const closeChartBtn = document.getElementById("close-chart-btn");
const chartSelect1 = document.getElementById("chart-select-1");
const chartSelect2 = document.getElementById("chart-select-2");
const showChartBtn = document.getElementById("show-chart-btn");
const skeletonOnlyBtn = document.getElementById("skeleton-only-btn");
const chartRangeStart = document.getElementById("chart-range-start");
const chartRangeEnd = document.getElementById("chart-range-end");
const rangeStartVal = document.getElementById("range-start-val");
const rangeEndVal = document.getElementById("range-end-val");

// App State
let poseLandmarker = undefined;
let webcamRunning = false;
let isVideoMode = false;
let isImageMode = false;
let skeletonOnlyMode = false;
let currentFacingMode = "user"; // "user" or "environment"
let localVideoTrack = null;
let lastVideoTime = -1;
let drawingUtils = null;
let frameCount = 0;
let lastFpsUpdate = 0;

// Recording State
let mediaRecorder = null;
let recordedBlobs = [];

// 時系列角度解析ヒストリー
let analysisHistory = []; // Array of { time: number, angles: { [mode]: { left, right } } }
let jointChart = null;

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
    
    const spine = vec.normalize(vec.sub(shoulderCenter, hipCenter));
    const shoulderAxis = vec.normalize(vec.sub(L_sh, R_sh));
    const forward = vec.normalize(vec.cross(shoulderAxis, spine));
    
    return { spine, shoulderAxis, forward };
};

// 関節運動の計算
const computeJointAngles = (landmarks, mode) => {
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

        if (!sh || !el || !wr || !hp || !kn || !ak || !idx || !ft) return null;

        switch (mode) {
            case "elbow_flexion": {
                const arm = vec.sub(sh, el);
                const forearm = vec.sub(wr, el);
                return vec.angle(arm, forearm);
            }
            case "knee_flexion": {
                const thigh = vec.sub(hp, kn);
                const calf = vec.sub(ak, kn);
                return vec.angle(thigh, calf);
            }
            case "wrist_flexion": {
                const forearm = vec.sub(el, wr);
                const hand = vec.sub(idx, wr);
                return vec.angle(forearm, hand);
            }
            case "ankle_flexion": {
                const calf = vec.sub(kn, ak);
                const foot = vec.sub(ft, ak);
                return vec.angle(calf, foot);
            }
            case "shoulder_flexion": {
                const armVec = vec.sub(el, sh);
                const planeAxis = body.shoulderAxis;
                const armProj = vec.normalize(vec.sub(armVec, vec.multiplyScalar(planeAxis, vec.dot(armVec, planeAxis))));
                const angle = vec.angle(trunkDown, armProj);
                const isFlexion = vec.dot(armProj, body.forward) > 0;
                return isFlexion ? `屈曲 ${angle}` : `伸展 ${angle}`;
            }
            case "shoulder_abduction": {
                const armVec = vec.sub(el, sh);
                const planeAxis = body.forward;
                const armProj = vec.normalize(vec.sub(armVec, vec.multiplyScalar(planeAxis, vec.dot(armVec, planeAxis))));
                const angle = vec.angle(trunkDown, armProj);
                const isAbduction = isLeft ? (vec.dot(armProj, body.shoulderAxis) > 0) : (vec.dot(armProj, body.shoulderAxis) < 0);
                return isAbduction ? `外転 ${angle}` : `内転 ${angle}`;
            }
            case "shoulder_rotation": {
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
                const thighVec = vec.sub(kn, hp);
                const planeAxis = body.shoulderAxis;
                const thighProj = vec.normalize(vec.sub(thighVec, vec.multiplyScalar(planeAxis, vec.dot(thighVec, planeAxis))));
                const angle = vec.angle(trunkDown, thighProj);
                const isFlexion = vec.dot(thighProj, body.forward) > 0;
                return isFlexion ? `屈曲 ${angle}` : `伸展 ${angle}`;
            }
            case "hip_abduction": {
                const thighVec = vec.sub(kn, hp);
                const planeAxis = body.forward;
                const thighProj = vec.normalize(vec.sub(thighVec, vec.multiplyScalar(planeAxis, vec.dot(thighVec, planeAxis))));
                const angle = vec.angle(trunkDown, thighProj);
                const isAbduction = isLeft ? (vec.dot(thighProj, body.shoulderAxis) > 0) : (vec.dot(thighProj, body.shoulderAxis) < 0);
                return isAbduction ? `外転 ${angle}` : `内転 ${angle}`;
            }
            case "hip_rotation": {
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
            return `${direction} ${num}°`;
        }
        return `${val}°`;
    };

    leftAngleVal.innerText = formatValue(angles.left);
    rightAngleVal.innerText = formatValue(angles.right);
};

// ミラーリング（反転）状態の自動更新
const updateMirrorMode = () => {
    const videoWrapper = video.parentElement;
    if (webcamRunning && currentFacingMode === "user" && !isVideoMode && !isImageMode) {
        videoWrapper.classList.add("mirror");
    } else {
        videoWrapper.classList.remove("mirror");
    }
};

// 時系列データの保存
const saveToHistory = (time, landmarks) => {
    // 重複時間をスキップ
    if (analysisHistory.length > 0 && analysisHistory[analysisHistory.length - 1].time === time) {
        return;
    }

    const modes = [
        "elbow_flexion", "knee_flexion", "wrist_flexion", "ankle_flexion",
        "shoulder_flexion", "shoulder_abduction", "shoulder_rotation",
        "hip_flexion", "hip_abduction", "hip_rotation"
    ];

    const frameData = {
        time: Math.round(time * 100) / 100,
        angles: {}
    };

    modes.forEach(m => {
        const result = computeJointAngles(landmarks, m);
        const parseNum = (val) => {
            if (val === null) return null;
            if (typeof val === "string") {
                const parts = val.split(" ");
                return parseFloat(parts[parts.length - 1]);
            }
            return val;
        };

        frameData.angles[m] = {
            left: parseNum(result.left),
            right: parseNum(result.right)
        };
    });

    analysisHistory.push(frameData);
};

// リアルタイム骨格検出ループ (事後解析連動・クラッシュ防止＆フレームレート最適化版)
const predictWebcam = async () => {
    if (!webcamRunning) return;

    // フリーズ・骨格フリーズ防止: 動画一時停止中、バッファ待ち、または終了時は姿勢推定を行わず、描画ループだけを維持して待機
    // これにより、スマホの通信ラグ等で自動的にpause/waiting状態になった際にAIループが永久停止してしまうバグを完全に防ぎます
    const isVideoPaused = isVideoMode && (video.paused || video.ended);
    if (isVideoPaused) {
        if (webcamRunning) {
            window.requestAnimationFrame(predictWebcam);
        }
        return;
    }

    if (video.readyState < 2 || video.videoWidth === 0 || video.videoHeight === 0) {
        if (webcamRunning) {
            window.requestAnimationFrame(predictWebcam);
        }
        return;
    }

    if (canvasElement.width !== video.videoWidth || canvasElement.height !== video.videoHeight) {
        canvasElement.width = video.videoWidth;
        canvasElement.height = video.videoHeight;
    }

    const startTimeMs = performance.now();

    // 映像データが更新されたフレームのみ、またはカメラプレビュー時は毎フレーム姿勢推定を実行
    const isNewFrame = isVideoMode ? (video.currentTime !== lastVideoTime) : true;
    if (isNewFrame) {
        if (isVideoMode) {
            lastVideoTime = video.currentTime;
        }
        
        try {
            // MediaPipe Tasks-Vision は動画解析時、ミリ秒単位で単調増加するタイムスタンプを要求します
            const timestamp = isVideoMode ? Math.round(video.currentTime * 1000) : startTimeMs;
            const results = poseLandmarker.detectForVideo(video, timestamp);
            canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

            // 骨格のみモードの時は背景を黒で塗りつぶす
            if (skeletonOnlyMode) {
                canvasCtx.fillStyle = "#0d0e15";
                canvasCtx.fillRect(0, 0, canvasElement.width, canvasElement.height);
            }

            if (results.landmarks && results.landmarks.length > 0) {
                poseDetectedVal.innerText = isVideoMode ? "分析中" : "プレビュー中";
                poseDetectedVal.classList.add("active");
                
                const landmarks = results.landmarks[0];
                
                // 骨格接続線の描画 (※事後解析中または通常カメラ稼働中のみ描画する)
                // 録画中は、プレビューを軽くするため＆本来の画角チェックのために描画を非表示にする
                const isRecording = mediaRecorder && mediaRecorder.state === "recording";
                if (!isRecording || isVideoMode) {
                    drawingUtils.drawConnectors(
                        landmarks, 
                        PoseLandmarker.POSE_CONNECTIONS, 
                        { color: "#9d00ff", lineWidth: 4 }
                    );
                    drawingUtils.drawLandmarks(
                        landmarks, 
                        { color: "#00f0ff", lineWidth: 2, radius: 4 }
                    );
                    
                    // 角度と時系列の更新
                    updateHudAngles(landmarks);
                    if (isVideoMode) {
                        saveToHistory(video.currentTime, landmarks);
                    }
                }
            } else {
                poseDetectedVal.innerText = isVideoMode ? "未検出" : "プレビュー中";
                if (!isVideoMode) {
                    poseDetectedVal.classList.add("active");
                } else {
                    poseDetectedVal.classList.remove("active");
                }
                leftAngleVal.innerText = "--°";
                rightAngleVal.innerText = "--°";
            }
        } catch (detectError) {
            console.warn("Detection error (ignored during transition):", detectError);
        }

        calculateFps(startTimeMs);
    }

    if (webcamRunning) {
        window.requestAnimationFrame(predictWebcam);
    }
};

// 実行モードの切り替え (IMAGE / VIDEO)
const setRunningMode = async (mode) => {
    if (!poseLandmarker) return;
    await poseLandmarker.setOptions({ runningMode: mode });
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
        poseDetectedVal.classList.remove("active");
        switchCameraBtn.disabled = true;
        recordBtn.disabled = true;
        
        // 録画中なら停止
        stopRecording();
    } else {
        // カメラ開始
        webcamRunning = true;
        isVideoMode = false;
        isImageMode = false;
        video.controls = false;
        
        await setRunningMode("VIDEO");
        
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
        recordBtn.disabled = false;
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
const handleVideoUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    stopMediaStream();
    webcamRunning = true;
    isVideoMode = true;
    isImageMode = false;
    updateMirrorMode();
    
    await setRunningMode("VIDEO");

    // グラフのリセットと非表示
    analysisHistory = [];
    chartSection.classList.remove("active");
    chartSection.style.display = "none";

    video.src = URL.createObjectURL(file);
    video.controls = true;
    
    statusDot.classList.add("active");
    toggleBtn.innerHTML = `
        <svg class="btn-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <polygon points="5 3 19 12 5 21 5 3"></polygon>
        </svg>
        <span>カメラ開始</span>
    `;
    toggleBtn.className = "panel-btn secondary-btn";
    switchCameraBtn.disabled = true;
    recordBtn.disabled = true;

    video.addEventListener("loadeddata", () => {
        video.play();
    }, { once: true });
};

// 静静画のアップロードと姿勢解析 (写真撮影 / ファイル読み込み)
const handleImageUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    stopMediaStream();
    webcamRunning = false;
    isVideoMode = false;
    isImageMode = true;
    updateMirrorMode();
    chartSection.classList.remove("active");
    chartSection.style.display = "none";

    loadingScreen.classList.remove("inactive");
    loadingStatus.innerText = "写真を解析中...";

    const reader = new FileReader();
    reader.onload = async (e) => {
        const img = new Image();
        img.onload = async () => {
            await setRunningMode("IMAGE");

            // キャンバスサイズを画像解像度に同期
            canvasElement.width = img.width;
            canvasElement.height = img.height;
            
            // 画像を描画
            if (skeletonOnlyMode) {
                canvasCtx.fillStyle = "#0d0e15";
                canvasCtx.fillRect(0, 0, canvasElement.width, canvasElement.height);
            } else {
                canvasCtx.drawImage(img, 0, 0, img.width, img.height);
            }
            
            try {
                // 画像から姿勢を検出
                const results = poseLandmarker.detect(img);
                
                loadingScreen.classList.add("inactive");

                if (results.landmarks && results.landmarks.length > 0) {
                    poseDetectedVal.innerText = "画像解析完了";
                    poseDetectedVal.classList.add("active");

                    const landmarks = results.landmarks[0];
                    
                    // 骨格描画
                    drawingUtils.drawConnectors(
                        landmarks, 
                        PoseLandmarker.POSE_CONNECTIONS, 
                        { color: "#9d00ff", lineWidth: 4 }
                    );
                    drawingUtils.drawLandmarks(
                        landmarks, 
                        { color: "#00f0ff", lineWidth: 2, radius: 4 }
                    );

                    // HUDの角度表示を即時更新
                    updateHudAngles(landmarks);
                } else {
                    poseDetectedVal.innerText = "未検出";
                    poseDetectedVal.classList.remove("active");
                    alert("姿勢を検出できませんでした。全身が写っている写真を使用してください。");
                }
            } catch (imageErr) {
                loadingScreen.classList.add("inactive");
                console.error("Image detection error:", imageErr);
                alert("写真の解析中にエラーが発生しました。");
            }
        };
        img.src = e.target.result;
    };
    reader.readAsDataURL(file);
};

// 録画機能の制御
const toggleRecording = () => {
    if (!webcamRunning || isVideoMode) return;

    const isRecording = mediaRecorder && mediaRecorder.state === "recording";
    if (isRecording) {
        stopRecording();
    } else {
        startRecording();
    }
};

// 録画の開始
const startRecording = () => {
    recordedBlobs = [];
    const stream = video.srcObject;
    if (!stream) return;

    let options = { mimeType: 'video/webm;codecs=vp9' };
    if (!MediaRecorder.isTypeSupported(options.mimeType)) {
        options = { mimeType: 'video/webm;codecs=vp8' };
        if (!MediaRecorder.isTypeSupported(options.mimeType)) {
            options = { mimeType: 'video/webm' };
            if (!MediaRecorder.isTypeSupported(options.mimeType)) {
                options = { mimeType: 'video/mp4' };
            }
        }
    }

    try {
        mediaRecorder = new MediaRecorder(stream, options);
    } catch (e) {
        console.error('Exception while creating MediaRecorder:', e);
        alert('このブラウザ/デバイスは録画に対応していません。');
        return;
    }

    mediaRecorder.ondataavailable = (event) => {
        if (event.data && event.data.size > 0) {
            recordedBlobs.push(event.data);
        }
    };

    mediaRecorder.onstop = () => {
        const superBuffer = new Blob(recordedBlobs, { type: 'video/webm' });
        const videoURL = window.URL.createObjectURL(superBuffer);
        
        // 録画データをセットして自動的に「事後解析」モードへ移行
        stopMediaStream();
        webcamRunning = true;
        isVideoMode = true;
        isImageMode = false;
        updateMirrorMode();
        
        // グラフと軌跡のリセット
        analysisHistory = [];
        chartSection.classList.remove("active");
        chartSection.style.display = "none";
        
        video.src = videoURL;
        video.controls = true;
        
        video.addEventListener("loadeddata", () => {
            video.play();
        }, { once: true });
    };

    mediaRecorder.start();
    recordBtn.classList.add("recording");
    document.getElementById("record-btn-text").innerText = "録画停止";
};

// 録画の停止
const stopRecording = () => {
    if (mediaRecorder && mediaRecorder.state !== "inactive") {
        mediaRecorder.stop();
    }
    recordBtn.classList.remove("recording");
    document.getElementById("record-btn-text").innerText = "録画開始";
};

// グラフ描画制御
const showChartSection = () => {
    try {
        if (analysisHistory.length === 0) {
            alert("時系列データがありません。動画を解析してください。");
            return;
        }

        if (typeof Chart === "undefined") {
            alert("グラフ描画ライブラリ (Chart.js) がロードされていません。インターネットの接続状態を確認するか、ページを再読み込みしてください。");
            return;
        }

        // インラインの display: none を解除してから、アニメーション用にディレイを入れて active クラスを適用
        chartSection.style.display = "block";
        setTimeout(() => {
            chartSection.classList.add("active");
            renderChart();
        }, 10);
    } catch (err) {
        alert("グラフ表示中にエラーが発生しました:\n" + err.message + "\n" + err.stack);
    }
};

const renderChart = () => {
    try {
        if (analysisHistory.length === 0) return;

        const canvas = document.getElementById('jointChart');
        if (!canvas) {
            alert("グラフ用のキャンバス要素が見つかりません。");
            return;
        }
        const ctx = canvas.getContext('2d');
        const select1Val = chartSelect1.value;
        const select2Val = chartSelect2.value;

        const minTime = parseFloat(chartRangeStart.value);
        const maxTime = parseFloat(chartRangeEnd.value);

        // 開始時間と終了時間の整合性を保つ
        if (minTime > maxTime) {
            chartRangeStart.value = maxTime;
        }

        const filteredHistory = analysisHistory.filter(d => d.time >= minTime && d.time <= maxTime);
        if (filteredHistory.length === 0) return;

        // ノイズ除去のための移動平均（スムージング）ヘルパー (5フレーム平均)
        const smoothData = (data, windowSize = 5) => {
            const smoothed = [];
            for (let i = 0; i < data.length; i++) {
                let sum = 0;
                let count = 0;
                for (let j = Math.max(0, i - windowSize + 1); j <= i; j++) {
                    const val = data[j];
                    if (val !== null && val !== undefined && !isNaN(val)) {
                        sum += val;
                        count++;
                    }
                }
                smoothed.push(count > 0 ? Math.round((sum / count) * 10) / 10 : null);
            }
            return smoothed;
        };

        const labels = filteredHistory.map(d => `${d.time}s`);
        const datasets = [];

        // グラフ 1 (実線) のデータを抽出して平滑化
        const rawLeftData1 = filteredHistory.map(d => d.angles[select1Val]?.left);
        const rawRightData1 = filteredHistory.map(d => d.angles[select1Val]?.right);
        const leftData1 = smoothData(rawLeftData1, 5);
        const rightData1 = smoothData(rawRightData1, 5);
        const label1 = chartSelect1.options[chartSelect1.selectedIndex].text;

        datasets.push({
            label: `${label1} (左)`,
            data: leftData1,
            borderColor: '#00f0ff',
            backgroundColor: 'rgba(0, 240, 255, 0.05)',
            borderWidth: 2,
            tension: 0.15,
            fill: false
        });
        datasets.push({
            label: `${label1} (右)`,
            data: rightData1,
            borderColor: '#00a0ff',
            backgroundColor: 'rgba(0, 160, 255, 0.05)',
            borderWidth: 2,
            tension: 0.15,
            fill: false
        });

        // グラフ 2 (破線) が選択されていれば抽出して平滑化
        if (select2Val !== 'none') {
            const rawLeftData2 = filteredHistory.map(d => d.angles[select2Val]?.left);
            const rawRightData2 = filteredHistory.map(d => d.angles[select2Val]?.right);
            const leftData2 = smoothData(rawLeftData2, 5);
            const rightData2 = smoothData(rawRightData2, 5);
            const label2 = chartSelect2.options[chartSelect2.selectedIndex].text;

            datasets.push({
                label: `${label2} (左)`,
                data: leftData2,
                borderColor: '#9d00ff',
                borderDash: [5, 5],
                backgroundColor: 'rgba(157, 0, 255, 0.05)',
                borderWidth: 2,
                tension: 0.15,
                fill: false
            });
            datasets.push({
                label: `${label2} (右)`,
                data: rightData2,
                borderColor: '#ff00d0',
                borderDash: [5, 5],
                backgroundColor: 'rgba(255, 0, 208, 0.05)',
                borderWidth: 2,
                tension: 0.15,
                fill: false
            });
        }

        if (jointChart) {
            jointChart.destroy();
        }

        jointChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: datasets
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        labels: { color: '#f0f3ff', font: { size: 10 } }
                    }
                },
                scales: {
                    x: {
                        grid: { color: 'rgba(255,255,255,0.05)' },
                        ticks: { color: '#8a8f9f', font: { size: 9 } }
                    },
                    y: {
                        grid: { color: 'rgba(255,255,255,0.05)' },
                        ticks: { color: '#8a8f9f', font: { size: 9 } },
                        min: 0,
                        max: 180
                    }
                }
            }
        });
    } catch (renderErr) {
        alert("グラフ描画中にエラーが発生しました:\n" + renderErr.message + "\n" + renderErr.stack);
    }
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
recordBtn.addEventListener("click", toggleRecording);
uploadVideoBtn.addEventListener("click", () => videoFileInput.click());
uploadImageBtn.addEventListener("click", () => imageFileInput.click());

videoFileInput.addEventListener("change", handleVideoUpload);
imageFileInput.addEventListener("change", handleImageUpload);

closeChartBtn.addEventListener("click", () => {
    chartSection.classList.remove("active");
    // スライドアニメーション(0.4s)完了後に display: none を適用して完全に消去
    setTimeout(() => {
        chartSection.style.display = "none";
    }, 400);
});

chartSelect1.addEventListener("change", renderChart);
chartSelect2.addEventListener("change", renderChart);
showChartBtn.addEventListener("click", showChartSection);

// グラフ表示時間範囲スライダーの入力イベント監視
const onRangeChange = () => {
    const minT = parseFloat(chartRangeStart.value);
    const maxT = parseFloat(chartRangeEnd.value);
    
    if (minT > maxT) {
        chartRangeStart.value = maxT;
    }
    
    rangeStartVal.innerText = parseFloat(chartRangeStart.value).toFixed(1);
    rangeEndVal.innerText = parseFloat(chartRangeEnd.value).toFixed(1);
    
    renderChart();
};

chartRangeStart.addEventListener("input", onRangeChange);
chartRangeEnd.addEventListener("input", onRangeChange);

// フリーズ防止: 以前の pause 連動はスマートフォンのロード待ち等でAIループが永久停止する原因になるため廃止
// 再生中は predictWebcam 内で video.paused を検知し安全に早期リターンする形式に改めました
video.addEventListener("play", () => {
    if (isVideoMode) {
        webcamRunning = true;
    }
});

video.addEventListener("ended", () => {
    if (isVideoMode) {
        showChartSection();
    }
});

skeletonOnlyBtn.addEventListener("click", () => {
    skeletonOnlyMode = !skeletonOnlyMode;
    skeletonOnlyBtn.classList.toggle("active-mode", skeletonOnlyMode);
    skeletonOnlyBtn.querySelector("span").innerText = skeletonOnlyMode ? "骨格のみ: ON" : "骨格のみ: OFF";
    
    // 静止画モードかつファイルが読み込まれている場合は再描画
    if (isImageMode && imageFileInput.files[0]) {
        const fileEvent = { target: { files: [imageFileInput.files[0]] } };
        handleImageUpload(fileEvent);
    }
});

// 関節切り替え時にも角度表示を即時クリア/再計算させる
jointSelect.addEventListener("change", () => {
    leftAngleVal.innerText = "--°";
    rightAngleVal.innerText = "--°";
    
    // 静止画モードの場合は、その場で即再計算して描画
    if (isImageMode && imageFileInput.files[0]) {
        // 再ロードをトリガー
        const fileEvent = { target: { files: [imageFileInput.files[0]] } };
        handleImageUpload(fileEvent);
    }
});

// Disable buttons initially until model is loaded
toggleBtn.disabled = true;

// Start initialization
initPoseLandmarker();
