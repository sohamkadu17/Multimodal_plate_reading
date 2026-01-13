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
    # Reduced configs for speed - only best performing ones
    configs = [
        "--oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 --psm 7",
        "--oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 --psm 6",
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
            # Early exit if high confidence found
            if best_conf > 0.85:
                break
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
    """Optimized plate detection with blur handling"""
    img = cv2.imread(image_path)
    if img is None:
        return "", None, 0.0

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect blur and apply deblurring if needed
    def detect_blur(image):
        """Laplacian variance method to detect blur"""
        return cv2.Laplacian(image, cv2.CV_64F).var()

    def deblur_wiener(image):
        """Simple Wiener deconvolution for motion blur"""
        try:
            kernel_size = 5
            kernel = np.zeros((kernel_size, kernel_size))
            kernel[int((kernel_size-1)/2), :] = np.ones(kernel_size)
            kernel = kernel / kernel_size
            dummy = np.copy(image)
            dummy = cv2.filter2D(dummy, -1, kernel)
            return cv2.addWeighted(image, 1.5, dummy, -0.5, 0)
        except:
            return image

    def unsharp_strong(image):
        """Strong unsharp mask for blur"""
        blurred = cv2.GaussianBlur(image, (0, 0), 3)
        return cv2.addWeighted(image, 2.0, blurred, -1.0, 0)

    # CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    gray_clahe = clahe.apply(gray)

    # Check if image is blurry
    blur_score = detect_blur(gray_clahe)
    is_blurry = blur_score < 100

    # Reduce variants for speed - focus on essential preprocessing
    variants = []
    if is_blurry:
        # More aggressive deblurring for blurry images
        deblurred = deblur_wiener(gray_clahe)
        variants.append(("deblur", deblurred))
        variants.append(("sharp", unsharp_strong(deblurred)))
        variants.append(("bilateral", cv2.bilateralFilter(deblurred, 9, 75, 75)))
    else:
        # Standard processing for clear images
        variants.append(("clahe", gray_clahe))
        variants.append(("bilateral", cv2.bilateralFilter(gray_clahe, 9, 75, 75)))

    # Single upscale only if needed
    variants.append(("upscale", cv2.resize(gray_clahe, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)))

    candidates = []
    seen_regions = set()

    for name, var in variants:
        try:
            proc = var if len(var.shape) == 2 else cv2.cvtColor(var, cv2.COLOR_BGR2GRAY)
            
            # Single threshold approach for speed
            _, thr = cv2.threshold(proc, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Edge detection
            edged = cv2.Canny(thr, 50, 200)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
            
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 800:  # Minimum area threshold
                    continue
                    
                x, y, ww, hh = cv2.boundingRect(cnt)
                aspect = float(ww) / (hh + 1e-6)
                
                # License plate aspect ratio check
                if 1.8 < aspect < 7.0 and ww > w * 0.10:
                    # Avoid duplicate regions
                    region_key = (x // 10, y // 10, ww // 10, hh // 10)
                    if region_key in seen_regions:
                        continue
                    seen_regions.add(region_key)
                    
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
                    
                    # Enhanced preprocessing for crop
                    crop_clahe = clahe.apply(crop_gray)
                    crop_resized = cv2.resize(crop_clahe, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                    
                    if is_blurry:
                        crop_resized = unsharp_strong(crop_resized)
                    
                    _, crop_thr = cv2.threshold(crop_resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    candidates.append((crop_thr, x0, y0, x1, y1))
                    
                    # Limit candidates for speed
                    if len(candidates) >= 10:
                        break
        except Exception as e:
            continue
        
        if len(candidates) >= 10:
            break

    # Fallback: full image OCR if no candidates
    if not candidates:
        for name, var in variants[:2]:  # Only try first 2 variants
            try:
                proc = var if len(var.shape) == 2 else cv2.cvtColor(var, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(proc, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                _, thr = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                if thr is not None and thr.size > 0:
                    candidates.append((thr, 0, 0, w, h))
            except Exception:
                continue

    best_text = ""
    best_conf = 0.0
    best_img = None

    # Try OCR on candidates
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
            # Early exit if very confident
            if best_conf > 0.9:
                break
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
    












