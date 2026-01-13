import os
import uuid
import datetime
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import pytesseract
import librosa
import soundfile as sf
import subprocess
import shutil

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
PLATE_DIR = os.path.join(BASE_DIR, "plates")

for d in (UPLOAD_DIR, PLATE_DIR, STATIC_DIR):
    os.makedirs(d, exist_ok=True)

ALLOWED_IMAGE = {"png", "jpg", "jpeg"}
ALLOWED_VIDEO = {"mp4", "mov", "avi", "mkv", "webm"}
ALLOWED_AUDIO = {"wav", "mp3", "m4a", "ogg"}
DEFAULT_METERS_PER_PIXEL = 0.05

app = Flask(__name__, static_folder=STATIC_DIR, static_url_path="")

def allowed(filename, allowed_set):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_set

def save_file(fs, subfolder):
    filename = secure_filename(fs.filename)
    ext = filename.rsplit(".", 1)[1] if "." in filename else ""
    unique = f"{uuid.uuid4().hex}.{ext}" if ext else uuid.uuid4().hex
    folder = os.path.join(UPLOAD_DIR, subfolder)
    os.makedirs(folder, exist_ok=True)
    abs_path = os.path.join(folder, unique)
    fs.save(abs_path)
    rel_url = f"/uploads/{subfolder}/{unique}"
    return rel_url, abs_path

def ocr_from_image_variants(img_gray):
    best_text = ""
    best_conf = 0.0
    configs = [
        "--oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 --psm 7",
        "--oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 --psm 6",
        "--oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 --psm 11",
        "--oem 1 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 --psm 7",
    ]
    try:
        for cfg in configs:
            data = pytesseract.image_to_data(img_gray, output_type=pytesseract.Output.DICT, config=cfg)
            texts = data.get("text", [])
            confs = data.get("conf", [])
            for t, c in zip(texts, confs):
                if not t or not t.strip():
                    continue
                try:
                    cval = float(c)
                except Exception:
                    cval = -1.0
                cleaned = "".join(ch for ch in t.upper() if ch.isalnum())
                if cleaned and cval >= 0 and (cval / 100.0) >= best_conf:
                    best_conf = cval / 100.0
                    best_text = cleaned
    except Exception:
        pass

    # optional EasyOCR fallback if installed
    try:
        import easyocr
        reader = easyocr.Reader(['en'], gpu=False)
        res = reader.readtext(img_gray, detail=1, paragraph=False)
        for entry in res:
            txt = (entry[1] or "").upper()
            cleaned = "".join(ch for ch in txt if ch.isalnum())
            conf = float(entry[2]) if len(entry) > 2 else 0.0
            if cleaned and conf >= best_conf:
                best_conf = conf
                best_text = cleaned
    except Exception:
        pass

    return best_text, float(best_conf)


def detect_plate_and_ocr(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return "", None, 0.0

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def unsharp(image):
        blurred = cv2.GaussianBlur(image, (3,3), 0)
        return cv2.addWeighted(image, 1.6, blurred, -0.6, 0)

    def denoise(image):
        try:
            return cv2.fastNlMeansDenoising(image, None, h=10, templateWindowSize=7, searchWindowSize=21)
        except Exception:
            return image

    def upscale(image, fx=2, fy=2):
        return cv2.resize(image, None, fx=fx, fy=fy, interpolation=cv2.INTER_CUBIC)

    variants = []
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    gray_clahe = clahe.apply(gray)
    variants.append(gray_clahe)
    variants.append(unsharp(gray_clahe))
    variants.append(denoise(gray_clahe))
    variants.append(upscale(gray_clahe, fx=2, fy=2))
    variants.append(upscale(gray_clahe, fx=3, fy=3))
    variants.append(upscale(unsharp(gray_clahe), fx=2, fy=2))

    candidates = []

    for var in variants:
        try:
            proc = var if len(var.shape) == 2 else cv2.cvtColor(var, cv2.COLOR_BGR2GRAY)
            # multiple thresholding attempts
            _, thr_otsu = cv2.threshold(proc, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            thr_adapt = cv2.adaptiveThreshold(proc,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
            morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
            thr_otsu = cv2.morphologyEx(thr_otsu, cv2.MORPH_CLOSE, morph_kernel)
            thr_adapt = cv2.morphologyEx(thr_adapt, cv2.MORPH_CLOSE, morph_kernel)
            for proc_variant in (proc, thr_otsu, thr_adapt):
                edged = cv2.Canny(proc_variant, 30, 150)
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
                closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
                contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area < 600:  # lower threshold to allow smaller/blurrier plates
                        continue
                    x, y, ww, hh = cv2.boundingRect(cnt)
                    aspect = float(ww) / (hh + 1e-6)
                    if 1.6 < aspect < 7.5 and ww > max(30, w * 0.08):
                        pad_y = int(hh * 0.3)
                        pad_x = int(ww * 0.08)
                        y0 = max(0, y - pad_y)
                        y1 = min(h, y + hh + pad_y)
                        x0 = max(0, x - pad_x)
                        x1 = min(w, x + ww + pad_x)
                        if y1 <= y0 or x1 <= x0:
                            continue
                        crop = img[y0:y1, x0:x1]
                        if crop is None or crop.size == 0:
                            continue
                        crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                        crop_up = upscale(crop_gray, fx=2, fy=2)
                        crop_sharp = unsharp(crop_up)
                        crop_denoise = denoise(crop_sharp)
                        _, crop_thr = cv2.threshold(crop_denoise, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                        candidates.append(crop_thr)
                        candidates.append(crop_denoise)
                        candidates.append(crop_sharp)
        except Exception:
            continue

    # fallback: try OCR on the full image variants if no contours produced candidates
    if not candidates:
        for var in variants:
            try:
                proc = var if len(var.shape) == 2 else cv2.cvtColor(var, cv2.COLOR_BGR2GRAY)
                proc_up = upscale(proc, fx=3, fy=3)
                proc_sharp = unsharp(proc_up)
                _, proc_thr = cv2.threshold(proc_sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                candidates.append(proc_thr)
                candidates.append(proc_sharp)
            except Exception:
                continue

    best_text = ""
    best_conf = 0.0
    best_img = None
    for cand in candidates:
        try:
            if cand is None or getattr(cand, "size", 0) == 0:
                continue
            text, conf = ocr_from_image_variants(cand)
            # prefer longer strings with reasonable confidence
            score = conf * (1.0 + 0.15 * len(text))
            if score > best_conf:
                best_conf = score
                best_text = text
                best_img = cand
        except Exception:
            continue

    if best_conf > 1.0:
        best_conf = min(1.0, best_conf)

    plate_filename = None
    if best_img is not None:
        try:
            plate_filename = f"plate_{uuid.uuid4().hex}.png"
            plate_path = os.path.join(PLATE_DIR, plate_filename)
            cv2.imwrite(plate_path, best_img)
        except Exception:
            plate_filename = None

    return (best_text or "").strip(), plate_filename, float(best_conf)


def detect_plate_and_ocr(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return "", None, 0.0

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    gray_clahe = clahe.apply(gray)

    def unsharp(image):
        blurred = cv2.GaussianBlur(image, (3,3), 0)
        return cv2.addWeighted(image, 1.5, blurred, -0.5, 0)

    def gamma_correction(image, gamma=1.0):
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)

    variants = []
    variants.append(("clahe", gray_clahe))
    variants.append(("unsharp", unsharp(gray_clahe)))
    variants.append(("bilateral", cv2.bilateralFilter(gray_clahe, 9, 75, 75)))
    variants.append(("gamma_up", gamma_correction(gray_clahe, 1.3)))
    variants.append(("gamma_down", gamma_correction(gray_clahe, 0.7)))
    variants.append(("resized", cv2.resize(gray_clahe, None, fx=2, fy=2, interpolation=cv2.INTER_LANCZOS4)))

    candidates = []

    for name, var in variants:
        try:
            proc = var if len(var.shape) == 2 else cv2.cvtColor(var, cv2.COLOR_BGR2GRAY)
            edged = cv2.Canny(proc, 50, 200)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
            closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 1200:
                    continue
                x, y, ww, hh = cv2.boundingRect(cnt)
                aspect = float(ww) / (hh + 1e-6)
                if 2.0 < aspect < 6.5 and ww > w * 0.12:
                    pad_y = int(hh * 0.25)
                    pad_x = int(ww * 0.06)
                    y0 = max(0, y - pad_y)
                    y1 = min(h, y + hh + pad_y)
                    x0 = max(0, x - pad_x)
                    x1 = min(w, x + ww + pad_x)
                    if y1 <= y0 or x1 <= x0:
                        continue
                    crop = img[y0:y1, x0:x1]
                    if crop is None or crop.size == 0:
                        continue
                    crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                    crop_resized = cv2.resize(crop_gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                    _, crop_thr = cv2.threshold(crop_resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    crop_unsharp = unsharp(crop_resized)
                    crop_bilat = cv2.bilateralFilter(crop_resized, 9, 75, 75)
                    candidates.append((crop_thr, x0, y0, x1, y1))
                    candidates.append((crop_unsharp, x0, y0, x1, y1))
                    candidates.append((crop_bilat, x0, y0, x1, y1))
        except Exception as e:
            print("detect_plate variant error:", str(e))
            continue

    if not candidates:
        for name, var in variants:
            try:
                try_img = var if len(var.shape) == 2 else cv2.cvtColor(var, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(try_img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                _, thr = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                if thr is not None and thr.size > 0:
                    candidates.append((thr, 0, 0, w, h))
            except Exception as e:
                print("global variant error:", str(e))
                continue

    best_text = ""
    best_conf = 0.0
    best_img = None

    for crop_img, x0, y0, x1, y1 in candidates:
        try:
            if crop_img is None or getattr(crop_img, "size", 0) == 0:
                continue
            text, conf = ocr_from_image_variants(crop_img)
            score = conf * (1.0 + 0.12 * len(text))
            if score > best_conf:
                best_conf = score
                best_text = text
                best_img = crop_img
        except Exception as e:
            print("ocr candidate error:", str(e))
            continue

    if best_conf > 1.0:
        best_conf = min(1.0, best_conf)

    plate_filename = None
    if best_img is not None:
        try:
            plate_filename = f"plate_{uuid.uuid4().hex}.png"
            plate_path = os.path.join(PLATE_DIR, plate_filename)
            cv2.imwrite(plate_path, best_img)
        except Exception as e:
            print("saving plate image error:", str(e))
            plate_filename = None

    return (best_text or "").strip(), plate_filename, float(best_conf)

def analyze_video_speed(video_abs_path, meters_per_pixel=DEFAULT_METERS_PER_PIXEL):
    cap = cv2.VideoCapture(video_abs_path)
    if not cap.isOpened():
        return {"error": "cannot_open_video"}
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    backsub = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=40, detectShadows=False)
    centroids = []
    frame_index = 0
    sample_step = max(1, int(max(1, frame_count // 60)))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_index % sample_step == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            fg = backsub.apply(gray)
            _, fgth = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
            fgth = cv2.morphologyEx(fgth, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(fgth, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            best = None
            best_area = 0
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > best_area and area > 300:
                    best_area = area
                    x, y, ww, hh = cv2.boundingRect(cnt)
                    cx = x + ww//2
                    cy = y + hh//2
                    best = (cx, cy)
            if best:
                time_sec = frame_index / float(fps) if fps > 0 else frame_index / 25.0
                centroids.append((best[0], best[1], time_sec))
        frame_index += 1
    cap.release()
    if len(centroids) < 2:
        return {"error": "no_motion_detected", "centroids_found": len(centroids)}
    distances = []
    times = []
    dx_total = 0
    dy_total = 0
    for a, b in zip(centroids[:-1], centroids[1:]):
        dx = b[0] - a[0]
        dy = b[1] - a[1]
        dx_total += dx
        dy_total += dy
        dt = b[2] - a[2] if (b[2] - a[2]) != 0 else 1.0 / fps
        dpx = np.sqrt(dx*dx + dy*dy)
        distances.append(dpx)
        times.append(dt)
    total_time = sum(times)
    if total_time <= 0 or sum(distances) == 0:
        return {"error": "motion_calc_error"}
    avg_px_per_sec = sum(distances) / total_time
    meters_per_sec = avg_px_per_sec * meters_per_pixel
    kmh = meters_per_sec * 3.6
    
    # Calculate direction based on overall movement
    direction = "unknown"
    angle = np.arctan2(dy_total, dx_total) * 180 / np.pi
    if -45 <= angle < 45:
        direction = "right"
    elif 45 <= angle < 135:
        direction = "down"
    elif angle >= 135 or angle < -135:
        direction = "left"
    else:
        direction = "up"
    
    return {
        "px_per_sec": round(float(avg_px_per_sec), 2),
        "meters_per_sec": round(float(meters_per_sec), 3),
        "speed_kmh": round(float(kmh), 2),
        "direction": direction,
        "frames_sampled": len(centroids),
        "meters_per_pixel_used": meters_per_pixel
    }

def analyze_audio_engine(wav_path):
    """
    Analyze audio to detect engine/vehicle sound characteristics.
    Returns dict with engine type and confidence.
    """
    try:
        y, sr = librosa.load(wav_path, sr=16000, mono=True)
        
        # Extract features
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
        
        # Compute statistics
        mfcc_mean = np.mean(mfcc, axis=1)
        spectral_mean = np.mean(spectral_centroid)
        zcr_mean = np.mean(zero_crossing_rate)
        
        # Simple heuristic classification
        engine_type = "unknown"
        confidence = 0.5
        
        if spectral_mean > 3000 and zcr_mean > 0.1:
            engine_type = "motorcycle"
            confidence = 0.75
        elif spectral_mean > 2000 and spectral_mean <= 3000:
            engine_type = "car"
            confidence = 0.7
        elif spectral_mean <= 2000:
            engine_type = "truck"
            confidence = 0.65
        
        return {
            "engine": engine_type,
            "confidence": round(float(confidence), 3),
            "spectral_centroid_mean": round(float(spectral_mean), 2),
            "zero_crossing_rate_mean": round(float(zcr_mean), 4)
        }
    except Exception as e:
        print("audio analysis error:", e)
        return {"engine": "unknown", "confidence": 0.0}

def convert_to_wav_with_ffmpeg(input_path, output_wav_path):
    """
    Convert an audio file to WAV using ffmpeg. Returns True on success.
    Requires ffmpeg in PATH.
    """
    try:
        # ensure ffmpeg exists
        if not shutil.which("ffmpeg"):
            raise RuntimeError("ffmpeg not found in PATH")
        cmd = ["ffmpeg", "-y", "-i", input_path, "-ar", "16000", "-ac", "1", output_wav_path]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except Exception as e:
        print("ffmpeg convert failed:", e)
        return False

@app.route("/", methods=["GET"])
def index():
    index_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_file):
        return send_from_directory(STATIC_DIR, "index.html")
    return "index.html not found - put your frontend files in the 'static' folder.", 404

@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    rel = filename.lstrip("/")
    return send_from_directory(UPLOAD_DIR, rel)

@app.route("/plates/<path:filename>")
def plate_file(filename):
    return send_from_directory(PLATE_DIR, filename)

@app.route("/analyze", methods=["POST"])
def analyze():
    resp = {
        "status": "ok",
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "notes": request.form.get("notes", "") or ""
    }
    if "image" not in request.files or request.files["image"].filename == "":
        return jsonify({"status": "error", "message": "image missing"}), 400
    image_fs = request.files["image"]
    if not allowed(image_fs.filename, ALLOWED_IMAGE):
        return jsonify({"status": "error", "message": "unsupported image type"}), 400
    rel_img, abs_img = save_file(image_fs, "images")
    resp["original_image"] = rel_img
    plate_text, plate_filename, plate_conf = detect_plate_and_ocr(abs_img)
    resp["plate"] = plate_text
    resp["plate_confidence"] = round(float(plate_conf), 3)
    resp["plate_image"] = f"/plates/{plate_filename}" if plate_filename else None
    if "video" in request.files and request.files["video"].filename != "":
        video_fs = request.files["video"]
        if allowed(video_fs.filename, ALLOWED_VIDEO):
            rel_video, abs_video = save_file(video_fs, "videos")
            resp["original_video"] = rel_video
            video_res = analyze_video_speed(abs_video, meters_per_pixel=float(request.form.get("meters_per_pixel", DEFAULT_METERS_PER_PIXEL)))
            resp["video_summary"] = video_res
            if "speed_kmh" in video_res:
                resp["speed_kmh"] = video_res["speed_kmh"]
            if "direction" in video_res:
                resp["direction"] = video_res["direction"]
        else:
            resp["video_error"] = "unsupported_format"
        # Audio analysis (optional)
    if "audio" in request.files and request.files["audio"].filename != "":
        audio_fs = request.files["audio"]
        if allowed(audio_fs.filename, ALLOWED_AUDIO):
            rel_audio, abs_audio = save_file(audio_fs, "audio")
            resp["original_audio"] = rel_audio
            # Ensure we have a WAV file for librosa/soundfile
            try:
                base, ext = os.path.splitext(abs_audio)
                wav_path = base + ".wav"
                ext = ext.lower()
                if ext == ".wav":
                    # Already WAV — use directly
                    audio_file_to_use = abs_audio
                else:
                    # Try ffmpeg conversion
                    ok = convert_to_wav_with_ffmpeg(abs_audio, wav_path)
                    if ok and os.path.exists(wav_path):
                        audio_file_to_use = wav_path
                    else:
                        # Try librosa direct load as fallback (may work if audioread/ffmpeg present)
                        audio_file_to_use = abs_audio
                # Load and (re)write to ensure consistent format
                y, sr = librosa.load(audio_file_to_use, sr=None, mono=True)
                # write to wav_path to ensure waveform file for analysis
                sf.write(wav_path, y, sr)
                audio_res = analyze_audio_engine(wav_path)

                resp["audio_summary"] = audio_res
                resp["engine"] = audio_res.get("engine")
                resp["engine_confidence"] = audio_res.get("confidence")
            except Exception as e:
                print("audio processing error:", e)
                resp["audio_error"] = str(e)
        else:
            resp["audio_error"] = "unsupported_format"
    resp["vehicle_type"] = ""
    if "direction" not in resp:
        resp["direction"] = ""
    resp["confidence"] = resp.get("plate_confidence", 0.0)
    return jsonify(resp)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
    












