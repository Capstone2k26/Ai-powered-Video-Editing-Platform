import os
import subprocess
import tempfile
import math

import cv2
import numpy as np


# ============================================================
# VIDIOAI - INTELLIGENT VIDEO HIGHLIGHT ANALYZER
# ============================================================
#
# Pipeline:
#
# Video
#   ↓
# Duration / FPS detection
#   ↓
# Whole-video visual sampling
#   ↓
# Motion analysis
#   ↓
# Scene-change analysis
#   ↓
# Visual activity analysis
#   ↓
# Audio analysis (if available)
#   ↓
# Optional Whisper speech analysis
#   ↓
# Candidate highlight generation
#   ↓
# Candidate scoring
#   ↓
# Remove overlapping highlights
#   ↓
# Return best moments
#
# Compatible with:
#
# from highlight_ai import analyze_video
#
# highlights = analyze_video(video_path, requested_clips)
# ============================================================


# ============================================================
# SETTINGS
# ============================================================

ANALYSIS_WIDTH = 320
ANALYSIS_HEIGHT = 180

DEFAULT_SAMPLE_RATE = 22050

# How much surrounding video belongs to a candidate.
BASE_HIGHLIGHT_WINDOW = 30.0

# Maximum number of analysis points.
MAX_ANALYSIS_POINTS = 600

# Audio extraction format.
AUDIO_SAMPLE_RATE = 22050


# ============================================================
# OPTIONAL WHISPER
# ============================================================

WHISPER_AVAILABLE = False
whisper = None

try:

    import whisper

    WHISPER_AVAILABLE = True

    print("✅ Whisper module detected.")

except Exception:

    print(
        "ℹ️ Whisper not available. "
        "Continuing with visual/audio analysis."
    )


# ============================================================
# BASIC UTILITIES
# ============================================================

def safe_float(value, default=0.0):

    try:
        return float(value)

    except Exception:

        return default


def clamp(value, minimum=0.0, maximum=1.0):

    return max(
        minimum,
        min(
            maximum,
            value
        )
    )


def normalize_array(values):

    if not values:

        return []

    array = np.asarray(
        values,
        dtype=np.float32
    )

    minimum = float(
        np.min(array)
    )

    maximum = float(
        np.max(array)
    )

    if maximum - minimum < 1e-9:

        return [
            0.0
            for _ in values
        ]

    normalized = (
        array - minimum
    ) / (
        maximum - minimum
    )

    return normalized.tolist()


# ============================================================
# RUN EXTERNAL COMMAND
# ============================================================

def run_command(command):

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    if result.returncode != 0:

        error = (
            result.stderr.strip()
            or result.stdout.strip()
            or "External command failed."
        )

        raise RuntimeError(error)

    return result


# ============================================================
# VIDEO INFORMATION
# ============================================================

def get_video_info(video_path):

    print()
    print("🎥 Reading video information...")

    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "stream=codec_type,width,height,r_frame_rate",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        video_path
    ]

    result = run_command(
        command
    )

    import json

    data = json.loads(
        result.stdout
    )

    duration = 0.0
    width = 0
    height = 0
    fps = 30.0

    # --------------------------------------------------------
    # Duration
    # --------------------------------------------------------

    try:

        duration = float(
            data.get(
                "format",
                {}
            ).get(
                "duration",
                0
            )
        )

    except Exception:

        duration = 0.0

    # --------------------------------------------------------
    # Video stream
    # --------------------------------------------------------

    streams = data.get(
        "streams",
        []
    )

    for stream in streams:

        if stream.get(
            "codec_type"
        ) != "video":

            continue

        width = int(
            stream.get(
                "width",
                0
            )
        )

        height = int(
            stream.get(
                "height",
                0
            )
        )

        frame_rate = stream.get(
            "r_frame_rate",
            "30/1"
        )

        try:

            numerator, denominator = (
                frame_rate.split("/")
            )

            fps = (
                float(numerator)
                /
                float(denominator)
            )

        except Exception:

            fps = 30.0

        break

    if fps <= 0:

        fps = 30.0

    if duration <= 0:

        raise Exception(
            "Could not determine video duration."
        )

    print(
        f"🎥 Duration: {duration:.2f} seconds"
    )

    print(
        f"🎞️ Resolution: {width}x{height}"
    )

    print(
        f"🎞️ FPS: {fps:.2f}"
    )

    return {
        "duration": duration,
        "width": width,
        "height": height,
        "fps": fps
    }


# ============================================================
# CHECK AUDIO
# ============================================================

def has_audio_track(video_path):

    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=codec_name",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        video_path
    ]

    try:

        result = run_command(
            command
        )

        return bool(
            result.stdout.strip()
        )

    except Exception:

        return False


# ============================================================
# AUDIO EXTRACTION
# ============================================================

def extract_audio(video_path):

    print()
    print("🔊 Preparing audio analysis...")

    if not has_audio_track(
        video_path
    ):

        print(
            "ℹ️ No usable audio track found."
        )

        return None

    temporary_file = tempfile.NamedTemporaryFile(
        suffix=".wav",
        delete=False
    )

    audio_path = temporary_file.name

    temporary_file.close()

    command = [
        "ffmpeg",
        "-y",

        "-i",
        video_path,

        "-vn",

        "-ac",
        "1",

        "-ar",
        str(AUDIO_SAMPLE_RATE),

        "-c:a",
        "pcm_s16le",

        audio_path
    ]

    try:

        run_command(
            command
        )

        if not os.path.exists(
            audio_path
        ):

            return None

        if os.path.getsize(
            audio_path
        ) < 1000:

            return None

        print(
            "✅ Audio extracted."
        )

        return audio_path

    except Exception as e:

        print(
            "⚠️ Audio extraction failed:"
        )

        print(
            str(e)
        )

        try:

            if os.path.exists(
                audio_path
            ):

                os.remove(
                    audio_path
                )

        except Exception:

            pass

        return None


# ============================================================
# AUDIO ANALYSIS
# ============================================================

def analyze_audio(audio_path):

    if not audio_path:

        return {}

    print()
    print("🔊 Analyzing audio...")

    try:

        import librosa

        audio, sample_rate = librosa.load(
            audio_path,
            sr=AUDIO_SAMPLE_RATE,
            mono=True
        )

    except Exception as e:

        print(
            "⚠️ Librosa audio analysis failed:"
        )

        print(
            repr(e)
        )

        return {}

    if len(audio) == 0:

        return {}

    # --------------------------------------------------------
    # RMS energy
    # --------------------------------------------------------

    try:

        rms = librosa.feature.rms(
            y=audio,
            frame_length=2048,
            hop_length=512
        )[0]

    except Exception:

        return {}

    if len(rms) == 0:

        return {}

    times = librosa.frames_to_time(
        np.arange(
            len(rms)
        ),
        sr=sample_rate,
        hop_length=512
    )

    normalized = normalize_array(
        rms.tolist()
    )

    audio_scores = {}

    for time, score in zip(
        times,
        normalized
    ):

        second = int(
            round(
                float(time)
            )
        )

        old = audio_scores.get(
            second,
            0.0
        )

        audio_scores[second] = max(
            old,
            float(score)
        )

    print(
        "✅ Audio analysis completed."
    )

    return audio_scores


# ============================================================
# WHISPER SPEECH ANALYSIS
# ============================================================

def analyze_speech(
    audio_path,
    duration
):

    if not WHISPER_AVAILABLE:

        return []

    if not audio_path:

        return []

    print()
    print(
        "🗣️ Whisper speech analysis..."
    )

    try:

        # ----------------------------------------------------
        # Small model for CPU-friendly processing.
        # ----------------------------------------------------

        model = whisper.load_model(
            "tiny"
        )

        result = model.transcribe(
            audio_path,
            fp16=False,
            verbose=False
        )

        segments = result.get(
            "segments",
            []
        )

        speech_segments = []

        for segment in segments:

            start = safe_float(
                segment.get(
                    "start",
                    0
                )
            )

            end = safe_float(
                segment.get(
                    "end",
                    start
                )
            )

            text = str(
                segment.get(
                    "text",
                    ""
                )
            ).strip()

            if not text:

                continue

            if end <= start:

                continue

            start = clamp(
                start,
                0,
                duration
            )

            end = clamp(
                end,
                0,
                duration
            )

            speech_segments.append({

                "start":
                    start,

                "end":
                    end,

                "text":
                    text,

                "importance":
                    calculate_speech_importance(
                        text
                    )
            })

        print(
            f"✅ Speech segments: "
            f"{len(speech_segments)}"
        )

        return speech_segments

    except Exception as e:

        print(
            "⚠️ Whisper analysis failed:"
        )

        print(
            repr(e)
        )

        print(
            "Continuing without Whisper."
        )

        return []


# ============================================================
# SPEECH IMPORTANCE
# ============================================================

def calculate_speech_importance(
    text
):

    if not text:

        return 0.0

    text_lower = text.lower()

    score = 0.0

    # --------------------------------------------------------
    # Excitement / emphasis words
    # --------------------------------------------------------

    strong_words = [
        "amazing",
        "incredible",
        "important",
        "secret",
        "best",
        "worst",
        "never",
        "always",
        "finally",
        "actually",
        "surprise",
        "surprising",
        "wow",
        "crazy",
        "shocking",
        "remember",
        "mistake",
        "problem",
        "solution",
        "why",
        "how",
        "what",
        "must",
        "special"
    ]

    for word in strong_words:

        if word in text_lower:

            score += 0.08

    # --------------------------------------------------------
    # Question
    # --------------------------------------------------------

    if "?" in text:

        score += 0.15

    # --------------------------------------------------------
    # Exclamation
    # --------------------------------------------------------

    if "!" in text:

        score += 0.15

    # --------------------------------------------------------
    # Sentence length
    # --------------------------------------------------------

    word_count = len(
        text.split()
    )

    if word_count >= 8:

        score += 0.15

    if word_count >= 15:

        score += 0.10

    return clamp(
        score
    )


# ============================================================
# GET SPEECH SCORE AT TIME
# ============================================================

def get_speech_score(
    speech_segments,
    time
):

    if not speech_segments:

        return 0.0

    best = 0.0

    for segment in speech_segments:

        if (
            segment["start"]
            <= time
            <= segment["end"]
        ):

            best = max(
                best,
                segment["importance"]
            )

    return clamp(
        best
    )


# ============================================================
# SAMPLE VIDEO
# ============================================================

def sample_video(
    video_path,
    duration,
    fps
):

    print()
    print(
        "👁️ Analyzing complete video..."
    )

    cap = cv2.VideoCapture(
        video_path
    )

    if not cap.isOpened():

        raise Exception(
            "Could not open video."
        )

    # --------------------------------------------------------
    # Determine sample interval
    # --------------------------------------------------------

    desired_points = int(
        min(
            MAX_ANALYSIS_POINTS,
            max(
                30,
                duration
            )
        )
    )

    interval = (
        duration /
        desired_points
    )

    interval = max(
        0.5,
        interval
    )

    print(
        f"📊 Analysis interval: "
        f"{interval:.2f} seconds"
    )

    print(
        f"📊 Approx analysis points: "
        f"{int(duration / interval)}"
    )

    # --------------------------------------------------------
    # Analysis data
    # --------------------------------------------------------

    results = []

    previous_gray = None
    previous_histogram = None

    current_time = 0.0

    while current_time < duration:

        cap.set(
            cv2.CAP_PROP_POS_MSEC,
            current_time * 1000
        )

        ret, frame = cap.read()

        if not ret:

            current_time += interval

            continue

        try:

            small = cv2.resize(
                frame,
                (
                    ANALYSIS_WIDTH,
                    ANALYSIS_HEIGHT
                )
            )

        except Exception:

            current_time += interval

            continue

        # ====================================================
        # GRAYSCALE
        # ====================================================

        gray = cv2.cvtColor(
            small,
            cv2.COLOR_BGR2GRAY
        )

        # ====================================================
        # MOTION
        # ====================================================

        motion = 0.0

        if previous_gray is not None:

            difference = cv2.absdiff(
                previous_gray,
                gray
            )

            motion = float(
                difference.mean()
            )

        motion_score = clamp(
            motion / 45.0
        )

        # ====================================================
        # SCENE CHANGE
        # ====================================================

        histogram = cv2.calcHist(
            [small],
            [0, 1],
            None,
            [32, 32],
            [
                0,
                256,
                0,
                256
            ]
        )

        cv2.normalize(
            histogram,
            histogram
        )

        scene_score = 0.0

        if previous_histogram is not None:

            difference = cv2.compareHist(
                previous_histogram,
                histogram,
                cv2.HISTCMP_BHATTACHARYYA
            )

            scene_score = clamp(
                float(difference)
            )

        # ====================================================
        # VISUAL CONTRAST
        # ====================================================

        contrast = float(
            gray.std()
        )

        contrast_score = clamp(
            contrast / 65.0
        )

        # ====================================================
        # EDGE / DETAIL
        # ====================================================

        edges = cv2.Canny(
            gray,
            80,
            160
        )

        edge_density = float(
            np.mean(
                edges > 0
            )
        )

        detail_score = clamp(
            edge_density / 0.25
        )

        # ====================================================
        # COLOR VARIATION
        # ====================================================

        hsv = cv2.cvtColor(
            small,
            cv2.COLOR_BGR2HSV
        )

        saturation = float(
            np.mean(
                hsv[:, :, 1]
            )
        )

        saturation_score = clamp(
            saturation / 180.0
        )

        # ====================================================
        # VISUAL SCORE
        # ====================================================

        visual_score = (

            motion_score * 0.40

            +

            scene_score * 0.25

            +

            contrast_score * 0.15

            +

            detail_score * 0.10

            +

            saturation_score * 0.10

        )

        visual_score = clamp(
            visual_score
        )

        results.append({

            "time":
                float(current_time),

            "motion":
                float(motion_score),

            "scene_change":
                float(scene_score),

            "contrast":
                float(contrast_score),

            "detail":
                float(detail_score),

            "saturation":
                float(saturation_score),

            "visual":
                float(visual_score)
        })

        previous_gray = gray
        previous_histogram = histogram

        current_time += interval

    cap.release()

    print(
        f"✅ Visual analysis points: "
        f"{len(results)}"
    )

    return results


# ============================================================
# ADD AUDIO SCORES
# ============================================================

def attach_audio_scores(
    results,
    audio_scores
):

    if not results:

        return results

    for item in results:

        second = int(
            round(
                item["time"]
            )
        )

        item["audio"] = float(
            audio_scores.get(
                second,
                0.0
            )
        )

    return results


# ============================================================
# ADD SPEECH SCORES
# ============================================================

def attach_speech_scores(
    results,
    speech_segments
):

    if not results:

        return results

    for item in results:

        item["speech"] = (
            get_speech_score(
                speech_segments,
                item["time"]
            )
        )

    return results


# ============================================================
# TEMPORAL SMOOTHING
# ============================================================

def smooth_scores(
    results
):

    if len(results) < 3:

        return results

    original = [
        item["visual"]
        for item in results
    ]

    for index, item in enumerate(
        results
    ):

        nearby = []

        for offset in [
            -2,
            -1,
            0,
            1,
            2
        ]:

            position = (
                index + offset
            )

            if (
                0
                <= position
                < len(original)
            ):

                nearby.append(
                    original[position]
                )

        if nearby:

            item["visual_smooth"] = float(
                np.mean(
                    nearby
                )
            )

        else:

            item["visual_smooth"] = (
                item["visual"]
            )

    return results


# ============================================================
# FINAL SCORE
# ============================================================

def calculate_final_scores(
    results
):

    for item in results:

        visual = item.get(
            "visual_smooth",
            item.get(
                "visual",
                0
            )
        )

        audio = item.get(
            "audio",
            0
        )

        speech = item.get(
            "speech",
            0
        )

        motion = item.get(
            "motion",
            0
        )

        scene = item.get(
            "scene_change",
            0
        )

        # ----------------------------------------------------
        # Combined intelligence score
        # ----------------------------------------------------

        score = (

            visual * 0.48

            +

            audio * 0.18

            +

            speech * 0.18

            +

            motion * 0.08

            +

            scene * 0.08

        )

        item["score"] = clamp(
            score
        )

    return results


# ============================================================
# CREATE HIGHLIGHT CANDIDATES
# ============================================================

def create_candidates(
    results,
    duration
):

    if not results:

        return []

    candidates = []

    # --------------------------------------------------------
    # Candidate window based on video length.
    #
    # This is only the region used for ranking.
    # app.py decides final Short duration.
    # --------------------------------------------------------

    window = min(
        BASE_HIGHLIGHT_WINDOW,
        duration
    )

    half_window = (
        window / 2.0
    )

    for item in results:

        center = float(
            item["time"]
        )

        start = max(
            0.0,
            center - half_window
        )

        end = min(
            duration,
            center + half_window
        )

        candidates.append({

            "start":
                start,

            "end":
                end,

            "time":
                center,

            "duration":
                end - start,

            "motion":
                item.get(
                    "motion",
                    0
                ),

            "scene_change":
                item.get(
                    "scene_change",
                    0
                ),

            "audio":
                item.get(
                    "audio",
                    0
                ),

            "speech":
                item.get(
                    "speech",
                    0
                ),

            "visual":
                item.get(
                    "visual",
                    0
                ),

            "score":
                item.get(
                    "score",
                    0
                )
        })

    return candidates


# ============================================================
# NON-MAXIMUM SUPPRESSION
# ============================================================

def remove_overlapping_candidates(
    candidates,
    requested_clips,
    duration
):

    if not candidates:

        return []

    ranked = sorted(
        candidates,
        key=lambda x:
            x["score"],
        reverse=True
    )

    selected = []

    # --------------------------------------------------------
    # Dynamic separation
    # --------------------------------------------------------

    separation = max(
        8.0,
        duration /
        (
            requested_clips * 2
        )
    )

    for candidate in ranked:

        candidate_time = (
            candidate["time"]
        )

        too_close = False

        for chosen in selected:

            if abs(
                candidate_time
                -
                chosen["time"]
            ) < separation:

                too_close = True

                break

        if too_close:

            continue

        selected.append(
            candidate
        )

        if len(selected) >= requested_clips:

            break

    # --------------------------------------------------------
    # Fallback
    # --------------------------------------------------------

    if len(selected) < requested_clips:

        for candidate in ranked:

            if candidate in selected:

                continue

            selected.append(
                candidate
            )

            if len(selected) >= requested_clips:

                break

    return selected[
        :requested_clips
    ]


# ============================================================
# NORMALIZE SCORES
# ============================================================

def normalize_final_scores(
    highlights
):

    if not highlights:

        return highlights

    maximum = max(
        item.get(
            "score",
            0
        )
        for item in highlights
    )

    minimum = min(
        item.get(
            "score",
            0
        )
        for item in highlights
    )

    difference = (
        maximum - minimum
    )

    if difference < 1e-9:

        for item in highlights:

            item["score"] = 1.0

        return highlights

    for item in highlights:

        item["score"] = (

            item["score"]
            -
            minimum

        ) / difference

    return highlights


# ============================================================
# ANALYZE VIDEO
# ============================================================

def analyze_video(
    video_path,
    requested_clips=3
):

    print()
    print("=" * 56)
    print("🧠 VIDIOAI INTELLIGENT VIDEO ANALYSIS")
    print("=" * 56)

    # ========================================================
    # VALIDATE
    # ========================================================

    if not os.path.exists(
        video_path
    ):

        raise FileNotFoundError(
            f"Video not found: {video_path}"
        )

    try:

        requested_clips = int(
            requested_clips
        )

    except Exception:

        requested_clips = 3

    if requested_clips not in [
        3,
        5
    ]:

        requested_clips = 3

    # ========================================================
    # VIDEO INFORMATION
    # ========================================================

    info = get_video_info(
        video_path
    )

    duration = info[
        "duration"
    ]

    fps = info[
        "fps"
    ]

    print(
        f"🎞️ FPS: {fps:.2f}"
    )

    # ========================================================
    # AUDIO
    # ========================================================

    audio_path = extract_audio(
        video_path
    )

    audio_scores = analyze_audio(
        audio_path
    )

    # ========================================================
    # SPEECH
    # ========================================================

    speech_segments = []

    if audio_path and WHISPER_AVAILABLE:

        speech_segments = analyze_speech(
            audio_path,
            duration
        )

    elif audio_path:

        print()
        print(
            "ℹ️ Whisper not installed."
        )

        print(
            "ℹ️ Continuing with audio + visual analysis."
        )

    # ========================================================
    # VISUAL ANALYSIS
    # ========================================================

    results = sample_video(
        video_path,
        duration,
        fps
    )

    if not results:

        if audio_path:

            try:

                os.remove(
                    audio_path
                )

            except Exception:

                pass

        print(
            "❌ No visual analysis results."
        )

        return []

    # ========================================================
    # ATTACH AUDIO
    # ========================================================

    results = attach_audio_scores(
        results,
        audio_scores
    )

    # ========================================================
    # ATTACH SPEECH
    # ========================================================

    results = attach_speech_scores(
        results,
        speech_segments
    )

    # ========================================================
    # SMOOTH VISUAL SCORES
    # ========================================================

    results = smooth_scores(
        results
    )

    # ========================================================
    # FINAL SCORE
    # ========================================================

    results = calculate_final_scores(
        results
    )

    # ========================================================
    # CANDIDATES
    # ========================================================

    candidates = create_candidates(
        results,
        duration
    )

    # ========================================================
    # SELECT BEST
    # ========================================================

    selected = remove_overlapping_candidates(
        candidates,
        requested_clips,
        duration
    )

    # ========================================================
    # NORMALIZE
    # ========================================================

    selected = normalize_final_scores(
        selected
    )

    # ========================================================
    # SORT BY TIME
    # ========================================================

    selected.sort(
        key=lambda x:
            x["time"]
    )

    # ========================================================
    # CLEAN OUTPUT
    # ========================================================

    final_highlights = []

    for item in selected:

        start = clamp(
            item["start"],
            0,
            duration
        )

        end = clamp(
            item["end"],
            0,
            duration
        )

        center = clamp(
            item["time"],
            0,
            duration
        )

        final_highlights.append({

            "start":
                round(
                    start,
                    2
                ),

            "end":
                round(
                    end,
                    2
                ),

            "time":
                round(
                    center,
                    2
                ),

            "duration":
                round(
                    end - start,
                    2
                ),

            "motion":
                round(
                    float(
                        item.get(
                            "motion",
                            0
                        )
                    ),
                    3
                ),

            "scene_change":
                round(
                    float(
                        item.get(
                            "scene_change",
                            0
                        )
                    ),
                    3
                ),

            "audio":
                round(
                    float(
                        item.get(
                            "audio",
                            0
                        )
                    ),
                    3
                ),

            "speech":
                round(
                    float(
                        item.get(
                            "speech",
                            0
                        )
                    ),
                    3
                ),

            "visual":
                round(
                    float(
                        item.get(
                            "visual",
                            0
                        )
                    ),
                    3
                ),

            "score":
                round(
                    float(
                        item.get(
                            "score",
                            0
                        )
                    ),
                    3
                )
        })

    # ========================================================
    # DELETE TEMP AUDIO
    # ========================================================

    if audio_path:

        try:

            os.remove(
                audio_path
            )

        except Exception:

            pass

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print()
    print("=" * 56)
    print("⭐ VIDIOAI BEST MOMENTS")
    print("-" * 56)

    for index, item in enumerate(
        final_highlights,
        start=1
    ):

        print(
            f"Short {index}: "
            f"{item['start']:.2f}s → "
            f"{item['end']:.2f}s | "
            f"Score: "
            f"{item['score']:.2f}"
        )

        print(
            f"   Visual: "
            f"{item['visual']:.2f} | "
            f"Motion: "
            f"{item['motion']:.2f} | "
            f"Scene: "
            f"{item['scene_change']:.2f} | "
            f"Audio: "
            f"{item['audio']:.2f} | "
            f"Speech: "
            f"{item['speech']:.2f}"
        )

    print("-" * 56)

    print(
        "✅ Selected",
        len(final_highlights),
        "best moments."
    )

    print("=" * 56)

    return final_highlights


# ============================================================
# DIRECT TEST
# ============================================================

if __name__ == "__main__":

    print()
    print(
        "VidioAI highlight_ai.py"
    )

    print(
        "This module is normally called from app.py."
    )

    print()
    print(
        "Usage:"
    )

    print(
        "from highlight_ai import analyze_video"
    )

    print(
        "highlights = analyze_video("
        "'video.mp4', 3)"
    )