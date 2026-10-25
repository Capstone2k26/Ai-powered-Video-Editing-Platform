from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

import os
import subprocess
import traceback
import uuid


# ============================================================
# IMPORT AI ANALYZER
# ============================================================

from highlight_ai import analyze_video


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)
CORS(app)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "outputs")
TEMP_FOLDER = os.path.join(BASE_DIR, "temp")


os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(TEMP_FOLDER, exist_ok=True)


app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["OUTPUT_FOLDER"] = OUTPUT_FOLDER
app.config["TEMP_FOLDER"] = TEMP_FOLDER


# ============================================================
# ALLOWED VIDEO EXTENSIONS
# ============================================================

ALLOWED_EXTENSIONS = {
    "mp4",
    "mov",
    "avi",
    "mkv",
    "webm",
    "m4v"
}


def allowed_file(filename):

    if "." not in filename:
        return False

    extension = filename.rsplit(".", 1)[1].lower()

    return extension in ALLOWED_EXTENSIONS


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():

    return jsonify({
        "success": True,
        "message": "VidioAI Backend is running!",
        "version": "2.0"
    })


# ============================================================
# RUN FFMPEG / FFPROBE SAFELY
# ============================================================

def run_command(command):

    print()
    print("▶ Running command:")
    print(" ".join(command))

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    if result.returncode != 0:

        error_message = (
            result.stderr.strip()
            or result.stdout.strip()
            or "External command failed."
        )

        raise RuntimeError(error_message)

    return result


# ============================================================
# GET VIDEO DURATION
# ============================================================

def get_video_duration(video_path):

    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        video_path
    ]

    result = run_command(command)

    try:

        duration = float(
            result.stdout.strip()
        )

    except Exception:

        raise Exception(
            "Could not determine video duration."
        )

    if duration <= 0:

        raise Exception(
            "Video duration is invalid."
        )

    return duration


# ============================================================
# CHECK AUDIO TRACK
# ============================================================

def has_audio(video_path):

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

        result = run_command(command)

        return bool(
            result.stdout.strip()
        )

    except Exception:

        return False


# ============================================================
# COMPRESS VIDEO FOR AI
# ============================================================

def compress_video(
    input_path,
    output_path
):

    print()
    print("========================================")
    print("📦 COMPRESSING VIDEO FOR AI")
    print("========================================")

    command = [
        "ffmpeg",
        "-y",

        "-i",
        input_path,

        # ------------------------------------
        # Reduce resolution for faster analysis
        # ------------------------------------

        "-vf",
        "scale=640:-2",

        # ------------------------------------
        # Video compression
        # ------------------------------------

        "-c:v",
        "libx264",

        "-preset",
        "veryfast",

        "-crf",
        "32",

        # ------------------------------------
        # Audio
        # ------------------------------------

        "-c:a",
        "aac",

        "-b:a",
        "96k",

        # ------------------------------------
        # MP4 optimization
        # ------------------------------------

        "-movflags",
        "+faststart",

        output_path
    ]

    try:

        run_command(command)

    except Exception as e:

        print()
        print("❌ COMPRESSION FAILED")
        print(str(e))

        raise Exception(
            "FFmpeg could not compress the video."
        ) from e

    if not os.path.exists(output_path):

        raise Exception(
            "Compressed video was not created."
        )

    size = os.path.getsize(
        output_path
    )

    print()
    print("✅ Compression completed!")
    print(
        "Compressed size:",
        round(size / 1024 / 1024, 2),
        "MB"
    )


# ============================================================
# CALCULATE SHORT DURATION
# ============================================================

def calculate_short_duration(
    total_duration,
    requested_clips
):

    """
    Adaptive Short duration.

    Example:

    10 minute video = 600 seconds

    3 Shorts:
        600 × 0.75 / 3
        = 150 seconds
        = 2.5 minutes

    5 Shorts:
        600 × 0.75 / 5
        = 90 seconds
        = 1.5 minutes

    This gives the AI enough room to include
    multiple important moments inside each Short.
    """

    if total_duration <= 0:
        raise Exception(
            "Invalid video duration."
        )

    if requested_clips <= 0:
        raise Exception(
            "Invalid number of Shorts."
        )

    usable_video_duration = (
        total_duration * 0.75
    )

    short_duration = (
        usable_video_duration /
        requested_clips
    )

    # ----------------------------------------
    # Safety limits
    # ----------------------------------------

    # Minimum
    short_duration = max(
        5.0,
        short_duration
    )

    # Maximum
    short_duration = min(
        300.0,
        short_duration
    )

    # Never exceed original video
    short_duration = min(
        short_duration,
        total_duration
    )

    return short_duration


# ============================================================
# GET HIGHLIGHT CENTER
# ============================================================

def get_highlight_center(
    highlight,
    total_duration
):

    if not isinstance(
        highlight,
        dict
    ):

        raise Exception(
            "Invalid AI highlight."
        )

    # ----------------------------------------
    # start + end
    # ----------------------------------------

    if (
        "start" in highlight
        and
        "end" in highlight
    ):

        start = float(
            highlight["start"]
        )

        end = float(
            highlight["end"]
        )

        if end < start:

            start, end = (
                end,
                start
            )

        center = (
            start + end
        ) / 2.0

    # ----------------------------------------
    # time
    # ----------------------------------------

    elif "time" in highlight:

        center = float(
            highlight["time"]
        )

    else:

        raise Exception(
            "AI highlight does not contain time information."
        )

    # ----------------------------------------
    # Safety
    # ----------------------------------------

    center = max(
        0.0,
        min(
            center,
            total_duration
        )
    )

    return center


# ============================================================
# CREATE AI SHORT
# ============================================================

def create_ai_short(
    input_path,
    output_path,
    highlight,
    total_duration,
    short_duration
):

    highlight_time = get_highlight_center(
        highlight,
        total_duration
    )

    # ----------------------------------------
    # Put AI highlight near center
    # ----------------------------------------

    start_time = (
        highlight_time
        -
        short_duration / 2.0
    )

    # ----------------------------------------
    # Keep inside video
    # ----------------------------------------

    start_time = max(
        0.0,
        start_time
    )

    max_start = max(
        0.0,
        total_duration - short_duration
    )

    start_time = min(
        start_time,
        max_start
    )

    end_time = min(
        total_duration,
        start_time + short_duration
    )

    actual_duration = (
        end_time - start_time
    )

    if actual_duration <= 0:

        raise Exception(
            "Invalid Short duration."
        )

    print()
    print("🎯 AI Highlight:")
    print(
        round(
            highlight_time,
            2
        ),
        "seconds"
    )

    print(
        "🎬 Start:",
        round(
            start_time,
            2
        )
    )

    print(
        "🎬 End:",
        round(
            end_time,
            2
        )
    )

    print(
        "⏱️ Duration:",
        round(
            actual_duration,
            2
        )
    )

    # ========================================================
    # VERTICAL 9:16 FORMAT
    # ========================================================

    vertical_filter = (
        "scale=720:1280:"
        "force_original_aspect_ratio=decrease,"
        "pad=720:1280:"
        "(ow-iw)/2:"
        "(oh-ih)/2"
    )

    command = [
        "ffmpeg",
        "-y",

        # ------------------------------------
        # Start point
        # ------------------------------------

        "-ss",
        f"{start_time:.3f}",

        "-i",
        input_path,

        # ------------------------------------
        # Duration
        # ------------------------------------

        "-t",
        f"{actual_duration:.3f}",

        # ------------------------------------
        # Vertical video
        # ------------------------------------

        "-vf",
        vertical_filter,

        # ------------------------------------
        # Video
        # ------------------------------------

        "-c:v",
        "libx264",

        "-preset",
        "veryfast",

        "-crf",
        "23",

        # ------------------------------------
        # Audio
        # ------------------------------------

        "-c:a",
        "aac",

        "-b:a",
        "128k",

        "-movflags",
        "+faststart",

        output_path
    ]

    print()
    print("🎬 Creating Short...")

    run_command(command)

    if not os.path.exists(
        output_path
    ):

        raise Exception(
            "Short file was not created."
        )

    return {

        "start":
            start_time,

        "end":
            end_time,

        "duration":
            actual_duration,

        "highlight_time":
            highlight_time
    }


# ============================================================
# SELECT UNIQUE AI HIGHLIGHTS
# ============================================================

def select_highlights(
    highlights,
    requested_count,
    total_duration
):

    if not highlights:

        return []

    # ----------------------------------------
    # Sort by AI score
    # ----------------------------------------

    candidates = sorted(
        highlights,
        key=lambda x:
            float(
                x.get(
                    "score",
                    0
                )
            ),
        reverse=True
    )

    selected = []

    # ----------------------------------------
    # Minimum separation
    # ----------------------------------------

    minimum_gap = max(
        5.0,
        total_duration /
        (
            requested_count * 2
        )
    )

    # ----------------------------------------
    # Select strongest different moments
    # ----------------------------------------

    for candidate in candidates:

        try:

            candidate_time = (
                get_highlight_center(
                    candidate,
                    total_duration
                )
            )

        except Exception:

            continue

        too_close = False

        for existing in selected:

            existing_time = (
                get_highlight_center(
                    existing,
                    total_duration
                )
            )

            if abs(
                candidate_time -
                existing_time
            ) < minimum_gap:

                too_close = True
                break

        if too_close:

            continue

        selected.append(
            candidate
        )

        if len(selected) >= requested_count:

            break

    # ----------------------------------------
    # Fallback
    # ----------------------------------------

    if len(selected) < requested_count:

        for candidate in candidates:

            if candidate in selected:

                continue

            selected.append(
                candidate
            )

            if len(selected) >= requested_count:

                break

    return selected[
        :requested_count
    ]


# ============================================================
# UPLOAD + AI PROCESSING
# ============================================================

@app.route(
    "/upload",
    methods=["POST"]
)
def upload_video():

    print()
    print("========================================")
    print("🚀 VIDIOAI NEW REQUEST")
    print("========================================")

    # ========================================================
    # CHECK FILE
    # ========================================================

    if "video" not in request.files:

        return jsonify({

            "success":
                False,

            "message":
                "No video received."

        }), 400

    video = request.files[
        "video"
    ]

    if not video.filename:

        return jsonify({

            "success":
                False,

            "message":
                "No video selected."

        }), 400

    # ========================================================
    # CHECK EXTENSION
    # ========================================================

    if not allowed_file(
        video.filename
    ):

        return jsonify({

            "success":
                False,

            "message":
                "Unsupported video format."

        }), 400

    # ========================================================
    # NUMBER OF SHORTS
    # ========================================================

    try:

        requested_clips = int(
            request.form.get(
                "clips",
                3
            )
        )

    except Exception:

        requested_clips = 3

    # ----------------------------------------
    # Only allow 3 or 5
    # ----------------------------------------

    if requested_clips not in [
        3,
        5
    ]:

        requested_clips = 3

    print(
        "Requested Shorts:",
        requested_clips
    )

    # ========================================================
    # UNIQUE FILE ID
    # ========================================================

    file_id = uuid.uuid4().hex[
        :10
    ]

    original_filename = secure_filename(
        video.filename
    )

    name_without_extension = os.path.splitext(
        original_filename
    )[0]

    safe_name = (
        name_without_extension
        + "_"
        + file_id
    )

    input_filename = (
        safe_name
        + ".mp4"
    )

    input_path = os.path.join(
        app.config[
            "UPLOAD_FOLDER"
        ],
        input_filename
    )

    # ========================================================
    # SAVE VIDEO
    # ========================================================

    try:

        video.save(
            input_path
        )

    except Exception as e:

        traceback.print_exc()

        return jsonify({

            "success":
                False,

            "message":
                "Could not save uploaded video.",

            "error":
                str(e)

        }), 500

    print(
        "✅ Original video saved:"
    )

    print(
        input_path
    )

    # ========================================================
    # GET ORIGINAL DURATION
    # ========================================================

    try:

        total_duration = (
            get_video_duration(
                input_path
            )
        )

    except Exception as e:

        traceback.print_exc()

        return jsonify({

            "success":
                False,

            "message":
                "Could not read video duration.",

            "error":
                str(e)

        }), 500

    print()
    print(
        "🎥 Original duration:",
        round(
            total_duration,
            2
        ),
        "seconds"
    )

    # ========================================================
    # CHECK AUDIO
    # ========================================================

    audio_available = has_audio(
        input_path
    )

    if audio_available:

        print(
            "🔊 Audio track: YES"
        )

    else:

        print(
            "ℹ️ Audio track: NO"
        )

    # ========================================================
    # CALCULATE SHORT DURATION
    # ========================================================

    short_duration = (
        calculate_short_duration(
            total_duration,
            requested_clips
        )
    )

    print()
    print(
        "📐 Target Short duration:",
        round(
            short_duration,
            2
        ),
        "seconds"
    )

    print(
        "📐 Target Short duration:",
        round(
            short_duration / 60,
            2
        ),
        "minutes"
    )

    # ========================================================
    # COMPRESSED AI FILE
    # ========================================================

    compressed_filename = (
        safe_name
        + "_AI_compressed.mp4"
    )

    compressed_path = os.path.join(
        app.config[
            "TEMP_FOLDER"
        ],
        compressed_filename
    )

    # ========================================================
    # COMPRESS
    # ========================================================

    try:

        compress_video(
            input_path,
            compressed_path
        )

    except Exception as e:

        traceback.print_exc()

        return jsonify({

            "success":
                False,

            "message":
                "Video compression failed.",

            "error":
                str(e)

        }), 500

    # ========================================================
    # AI ANALYSIS
    # ========================================================

    print()
    print("========================================")
    print("🧠 STARTING AI VIDEO ANALYSIS")
    print("========================================")

    try:

        highlights = analyze_video(
            compressed_path,
            requested_clips
        )

    except Exception as e:

        print()
        print(
            "❌ AI ANALYSIS FAILED"
        )

        traceback.print_exc()

        return jsonify({

            "success":
                False,

            "message":
                "AI video analysis failed.",

            "error":
                str(e)

        }), 500

    if not highlights:

        return jsonify({

            "success":
                False,

            "message":
                "AI could not find useful highlights."

        }), 500

    # ========================================================
    # SELECT BEST UNIQUE HIGHLIGHTS
    # ========================================================

    selected_highlights = (
        select_highlights(
            highlights,
            requested_clips,
            total_duration
        )
    )

    if not selected_highlights:

        return jsonify({

            "success":
                False,

            "message":
                "No usable AI highlights found."

        }), 500

    print()
    print(
        "⭐ SELECTED AI MOMENTS"
    )

    for index, highlight in enumerate(
        selected_highlights,
        start=1
    ):

        try:

            center = (
                get_highlight_center(
                    highlight,
                    total_duration
                )
            )

            score = float(
                highlight.get(
                    "score",
                    0
                )
            )

            print(
                f"Short {index}: "
                f"{center:.2f}s | "
                f"Score: {score:.3f}"
            )

        except Exception:

            print(
                f"Short {index}: "
                "Invalid highlight"
            )

    # ========================================================
    # CREATE SHORTS
    # ========================================================

    clips = []

    for index, highlight in enumerate(
        selected_highlights,
        start=1
    ):

        output_filename = (
            safe_name
            + f"_AI_short_{index}.mp4"
        )

        output_path = os.path.join(
            app.config[
                "OUTPUT_FOLDER"
            ],
            output_filename
        )

        print()
        print(
            "========================================"
        )

        print(
            f"🎬 CREATING SHORT {index}/{requested_clips}"
        )

        print(
            "========================================"
        )

        try:

            clip_info = (
                create_ai_short(
                    input_path=
                        input_path,

                    output_path=
                        output_path,

                    highlight=
                        highlight,

                    total_duration=
                        total_duration,

                    short_duration=
                        short_duration
                )
            )

        except Exception as e:

            print()
            print(
                "❌ SHORT CREATION FAILED"
            )

            traceback.print_exc()

            return jsonify({

                "success":
                    False,

                "message":
                    f"Could not create Short {index}.",

                "error":
                    str(e)

            }), 500

        score = float(
            highlight.get(
                "score",
                0
            )
        )

        clips.append({

            "clip_number":
                index,

            "filename":
                output_filename,

            "url":
                "/outputs/"
                + output_filename,

            "start":
                round(
                    clip_info[
                        "start"
                    ],
                    2
                ),

            "end":
                round(
                    clip_info[
                        "end"
                    ],
                    2
                ),

            "duration":
                round(
                    clip_info[
                        "duration"
                    ],
                    2
                ),

            "highlight_time":
                round(
                    clip_info[
                        "highlight_time"
                    ],
                    2
                ),

            "score":
                round(
                    score,
                    3
                )
        })

    # ========================================================
    # SUCCESS
    # ========================================================

    print()
    print("========================================")
    print("🎉 VIDIOAI COMPLETE")
    print("========================================")

    print(
        "Original duration:",
        round(
            total_duration,
            2
        ),
        "seconds"
    )

    print(
        "Shorts created:",
        len(clips)
    )

    print(
        "Short duration:",
        round(
            short_duration,
            2
        ),
        "seconds"
    )

    print(
        "========================================"
    )

    return jsonify({

        "success":
            True,

        "message":
            "VidioAI successfully analyzed the video and created Shorts.",

        "original_duration":
            round(
                total_duration,
                2
            ),

        "requested_clips":
            requested_clips,

        "created_clips":
            len(clips),

        "target_short_duration":
            round(
                short_duration,
                2
            ),

        "audio_available":
            audio_available,

        "clips":
            clips
    })


# ============================================================
# SERVE GENERATED SHORTS
# ============================================================

@app.route(
    "/outputs/<path:filename>"
)
def get_output(filename):

    return send_from_directory(
        app.config[
            "OUTPUT_FOLDER"
        ],
        filename
    )


# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":

    print()
    print("========================================")
    print("🚀 VIDIOAI BACKEND")
    print("========================================")
    print("Server: http://127.0.0.1:5000")
    print("========================================")

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )