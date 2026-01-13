// static/js/app.js
console.log("app.js loaded");

document.addEventListener("DOMContentLoaded", () => {
  const API_URL = "/analyze";

  // Elements
  const uploadForm = document.getElementById("uploadForm");
  const imageInput = document.getElementById("imageInput");
  const videoInput = document.getElementById("videoInput");
  const audioInput = document.getElementById("audioInput");
  const notes = document.getElementById("notes");

  const analyzeBtn = document.getElementById("analyzeBtn");
  const resetBtn = document.getElementById("resetBtn");
  const downloadBtn = document.getElementById("downloadBtn");

  const imgPreview = document.getElementById("imgPreview");
  const videoPreview = document.getElementById("videoPreview");
  const audioPreview = document.getElementById("audioPreview");

  const loading = document.getElementById("loading");
  const results = document.getElementById("results");
  const rawJson = document.getElementById("rawJson");

  const openCameraBtn = document.getElementById("openCameraBtn");
  const cameraModal = document.getElementById("cameraModal");
  const cameraStream = document.getElementById("cameraStream");
  const captureBtn = document.getElementById("captureBtn");
  const closeCameraBtn = document.getElementById("closeCameraBtn");

  const recordVideoBtn = document.getElementById("recordVideoBtn");
  const stopVideoBtn = document.getElementById("stopVideoBtn");
  const recordAudioBtn = document.getElementById("recordAudioBtn");
  const stopAudioBtn = document.getElementById("stopAudioBtn");

  // Defensive checks
  if (!analyzeBtn || !resetBtn || !imageInput) {
    console.error("Essential DOM elements missing. Check IDs in HTML.");
    return;
  }

  // Utility to safely set preview object URLs
  let lastObjectUrl = null;
  function setPreview(el, fileOrBlob) {
    try {
      if (lastObjectUrl) {
        URL.revokeObjectURL(lastObjectUrl);
        lastObjectUrl = null;
      }
      const url = URL.createObjectURL(fileOrBlob);
      lastObjectUrl = url;
      el.src = url;
      el.style.display = "block";
      console.log("Preview set:", el.id, fileOrBlob.name || fileOrBlob.type, fileOrBlob.size);
    } catch (e) {
      console.error("setPreview error:", e);
    }
  }

  // Reset form

   function clearForm() {
    console.warn("!! resetAll() was called !!");
    console.trace();
    document.getElementById('uploadForm').reset();

    imgPreview.style.display = 'none';
    imgPreview.src = '';

    videoPreview.style.display = 'none';
    videoPreview.src = '';

    audioPreview.style.display = 'none';
    audioPreview.src = '';

    results.innerHTML = '<p class="text-muted">No analysis performed yet.</p>';
    rawJson.textContent = '{}';

    downloadBtn.style.display = 'none';
  }

  resetBtn.addEventListener('click', clearForm);

  // function clearForm() {
  //   try {
  //     uploadForm.reset();
  //     imgPreview.style.display = "none"; imgPreview.src = "";
  //     videoPreview.style.display = "none"; videoPreview.src = "";
  //     audioPreview.style.display = "none"; audioPreview.src = "";
  //     results.innerHTML = '<p class="text-muted">No analysis performed yet.</p>';
  //     rawJson.textContent = "{}";
  //     downloadBtn.style.display = "none";
  //     console.log("Form reset");
  //   } catch (e) {
  //     console.error("clearForm error:", e);
  //   }
  // }
  // resetBtn.addEventListener("click", (e) => {
  //   e.preventDefault();
  //   clearForm();
  // });

  // Native previews when user selects files
  imageInput.addEventListener("change", () => {
    const f = imageInput.files[0];
    if (!f) { imgPreview.style.display = "none"; return; }
    if (!f.type.startsWith("image/")) { alert("Please select an image file."); imageInput.value = ""; return; }
    setPreview(imgPreview, f);
  });

  videoInput.addEventListener("change", () => {
    const f = videoInput.files[0];
    if (!f) { videoPreview.style.display = "none"; return; }
    if (!f.type.startsWith("video/")) { alert("Please select a video file."); videoInput.value = ""; return; }
    setPreview(videoPreview, f);
  });

  audioInput.addEventListener("change", () => {
    const f = audioInput.files[0];
    if (!f) { audioPreview.style.display = "none"; return; }
    if (!f.type.startsWith("audio/")) { alert("Please select an audio file."); audioInput.value = ""; return; }
    setPreview(audioPreview, f);
  });

  // ANALYZE handler
   analyzeBtn.addEventListener('click', async (e) => {
    e.preventDefault();
    if (!imageInput.files[0]) {
      alert("Please upload a vehicle image (required).");
      return;
    }

    const fd = new FormData();
    fd.append("image", imageInput.files[0]);
    if (videoInput.files[0]) fd.append("video", videoInput.files[0]);
    if (audioInput.files[0]) fd.append("audio", audioInput.files[0]);

    fd.append("notes", notes.value || "");

    analyzeBtn.disabled = true;
    loading.style.display = "flex";
    results.innerHTML = `<p class="text-muted">Sending files to server...</p>`;

    try {
      const resp = await fetch("/analyze", {
      method: "POST",
      body: fd,
});

    console.log("1. Fetch request completed. Status:",resp.status);


      if (!resp.ok) {
        const txt = await resp.text();
        throw new Error(`Server returned ${resp.status}: ${txt}`);
      }

      const data = await resp.json();
      displayResults(data);
      document.body.style.border= '10px solid limegreen';
      console. log("SUCCESS HANDLER COMPLETED!");

      rawJson.textContent = JSON.stringify(data, null, 2);

      const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
      downloadBtn.href = URL.createObjectURL(blob);
      downloadBtn.style.display = "inline-block";

    } catch (err) {
      console.error(err);
      results.innerHTML = `<div class="alert alert-danger py-2">Error: ${err.message}</div>`;
      rawJson.textContent = "{}";
      downloadBtn.style.display = "none";

    } finally {
      analyzeBtn.disabled = false;
      loading.style.display = "none";
    }
  });

   function displayResults(data) {
    const html = `
      <div><strong>Plate:</strong> ${data.plate || data.plate_number || '—'}</div>
      <div><strong>Engine:</strong> ${data.engine || data.engine_type || '—'}</div>
      <div><strong>Speed:</strong> ${data.speed_kmh ? data.speed_kmh + " km/h" : "—"}</div>
      <div><strong>Direction:</strong> ${data.direction || '—'}</div>
      <div><strong>Confidence:</strong> ${data.confidence ? (data.confidence * 100).toFixed(1) + "%" : "—"}</div>
    `;

    results.innerHTML = html;
  }


  // ---------- Camera capture ----------
  let camStream = null;
  openCameraBtn?.addEventListener("click", async () => {
    cameraModal.style.display = "flex";
    try {
      camStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" }, audio: false });
      cameraStream.srcObject = camStream;
      cameraStream.addEventListener("loadedmetadata", () => {
        cameraStream.play().catch(()=>{});
      }, { once: true });
    } catch (err) {
      console.error("Camera open error:", err);
      alert("Cannot access camera: " + (err.message || err));
      cameraModal.style.display = "none";
    }
  });

  captureBtn?.addEventListener("click", () => {
    if (!cameraStream || !cameraStream.videoWidth) {
      alert("Camera not ready yet. Wait a second and try again.");
      return;
    }
    const canvas = document.createElement("canvas");
    canvas.width = cameraStream.videoWidth;
    canvas.height = cameraStream.videoHeight;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(cameraStream, 0, 0, canvas.width, canvas.height);
    canvas.toBlob((blob) => {
      if (!blob) { alert("Capture failed"); return; }
      const file = new File([blob], "captured.jpg", { type: "image/jpeg" });
      const dt = new DataTransfer(); dt.items.add(file); imageInput.files = dt.files;
      imageInput.dispatchEvent(new Event("change"));
      setPreview(imgPreview, file);
      stopCamera();
      cameraModal.style.display = "none";
    }, "image/jpeg", 0.92);
  });

  closeCameraBtn?.addEventListener("click", () => {
    stopCamera();
    cameraModal.style.display = "none";
  });

  function stopCamera() {
    if (!camStream) return;
    camStream.getTracks().forEach(t => t.stop());
    camStream = null;
    cameraStream.srcObject = null;
    console.log("Camera stopped");
  }

  // ---------- Video recording ----------
  let mediaRecorderVideo = null;
  let videoChunks = [];
  function chooseRecordingMime(candidates) {
    for (const c of candidates) if (MediaRecorder.isTypeSupported && MediaRecorder.isTypeSupported(c)) return c;
    return "";
  }
  recordVideoBtn?.addEventListener("click", async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
      const mime = chooseRecordingMime(["video/webm;codecs=vp9,opus","video/webm;codecs=vp8,opus","video/mp4"]);
      mediaRecorderVideo = new MediaRecorder(stream, mime ? { mimeType: mime } : undefined);
      videoChunks = [];
      mediaRecorderVideo.ondataavailable = e => { if (e.data && e.data.size) videoChunks.push(e.data); };
      mediaRecorderVideo.onstop = () => {
        const blob = new Blob(videoChunks, { type: videoChunks[0]?.type || "video/webm" });
        const ext = blob.type.includes("mp4") ? ".mp4" : ".webm";
        const file = new File([blob], `recorded_video${ext}`, { type: blob.type });
        const dt = new DataTransfer(); dt.items.add(file); videoInput.files = dt.files;
        setPreview(videoPreview, file);
        stream.getTracks().forEach(t => t.stop());
        console.log("Video recorded:", file.name, file.size);
      };
      mediaRecorderVideo.start();
      recordVideoBtn.style.display = "none";
      stopVideoBtn.style.display = "inline-block";
      console.log("Video recording started");
    } catch (err) {
      console.error("Video record error:", err);
      alert("Unable to record video: " + (err.message || err));
    }
  });

  stopVideoBtn?.addEventListener("click", () => {
    if (mediaRecorderVideo && mediaRecorderVideo.state !== "inactive") mediaRecorderVideo.stop();
    recordVideoBtn.style.display = "inline-block";
    stopVideoBtn.style.display = "none";
  });

  // ---------- Audio recording ----------
  let mediaRecorderAudio = null;
  let audioChunksLocal = [];
  recordAudioBtn?.addEventListener("click", async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mime = MediaRecorder.isTypeSupported && MediaRecorder.isTypeSupported("audio/webm") ? "audio/webm" : "";
      mediaRecorderAudio = new MediaRecorder(stream, mime ? { mimeType: mime } : undefined);
      audioChunksLocal = [];
      mediaRecorderAudio.ondataavailable = e => { if (e.data && e.data.size) audioChunksLocal.push(e.data); };
      mediaRecorderAudio.onstop = () => {
        const blob = new Blob(audioChunksLocal, { type: audioChunksLocal[0]?.type || "audio/webm" });
        const ext = blob.type.includes("mpeg") || blob.type.includes("mp3") ? ".mp3" : ".webm";
        const file = new File([blob], `recorded_audio${ext}`, { type: blob.type });
        const dt = new DataTransfer(); dt.items.add(file); audioInput.files = dt.files;
        setPreview(audioPreview, file);
        stream.getTracks().forEach(t => t.stop());
        console.log("Audio recorded:", file.name, file.size);
      };
      mediaRecorderAudio.start();
      recordAudioBtn.style.display = "none";
      stopAudioBtn.style.display = "inline-block";
    } catch (err) {
      console.error("Audio record error:", err);
      alert("Unable to record audio: " + (err.message || err));
    }
  });

  stopAudioBtn?.addEventListener("click", () => {
    if (mediaRecorderAudio && mediaRecorderAudio.state !== "inactive") mediaRecorderAudio.stop();
    recordAudioBtn.style.display = "inline-block";
    stopAudioBtn.style.display = "none";
  });

  // Debug logs when files change
  [imageInput, videoInput, audioInput].forEach(input => {
    input?.addEventListener("change", () => {
      const f = input.files[0];
      console.log(input.id, "changed:", f ? { name: f.name, size: f.size, type: f.type } : null);
    });
  });

  console.log("app.js initialization complete");
});