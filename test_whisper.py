import os
import sys
import subprocess
import traceback


# ============================================================
# VIDIOAI WHISPER TEST
# ============================================================

print("=" * 60)
print("🎙️ VIDIOAI WHISPER AI TEST")
print("=" * 60)


# ============================================================
# CHECK PYTHON
# ============================================================

print()
print("🐍 Python:")
print(sys.version)


# ============================================================
# CHECK TORCH
# ============================================================

print()
print("🔥 Checking PyTorch...")

try:
    import torch

    print("✅ PyTorch imported successfully")
    print("   Version:", torch.__version__)

    try:
        print("   CUDA available:", torch.cuda.is_available())

        if torch.cuda.is_available():
            print("   GPU:", torch.cuda.get_device_name(0))
        else:
            print("   ℹ️ Using CPU")

    except Exception as e:
        print("⚠️ Could not check CUDA:", repr(e))

except Exception as e:

    print("❌ PyTorch could not be imported.")
    print()
    print("ERROR:")
    traceback.print_exc()

    sys.exit(1)


# ============================================================
# CHECK WHISPER
# ============================================================

print()
print("🎙️ Checking Whisper...")

try:
    import whisper

    print("✅ Whisper imported successfully")

except Exception as e:

    print("❌ Whisper could not be imported.")
    print()
    print("ERROR:")
    traceback.print_exc()

    print()
    print("Install Whisper with:")
    print("python -m pip install -U openai-whisper")

    sys.exit(1)


# ============================================================
# CHECK FFPROBE
# ============================================================

print()
print("🎬 Checking FFprobe...")

try:

    result = subprocess.run(
        [
            "ffprobe",
            "-version"
        ],
        capture_output=True,
        text=True
    )

    if result.returncode == 0:

        first_line = result.stdout.splitlines()[0]

        print("✅ FFprobe found")
        print("  ", first_line)

    else:

        print("❌ FFprobe returned an error.")
        sys.exit(1)

except FileNotFoundError:

    print("❌ FFprobe was not found.")
    print("Make sure FFmpeg is installed and added to PATH.")

    sys.exit(1)

except Exception as e:

    print("❌ FFprobe check failed:")
    print(repr(e))

    sys.exit(1)


# ============================================================
# FIND VIDEO
# ============================================================

print()
print("📁 Checking video...")


# You can change this filename.
VIDEO_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "test_whisper.mp4"
)


if not os.path.exists(VIDEO_PATH):

    print("⚠️ Test video was not found.")

    print()
    print("Expected location:")
    print(VIDEO_PATH)

    print()
    print("Put a video named:")
    print("test_whisper.mp4")

    print("inside:")
    print(os.path.dirname(os.path.abspath(__file__)))

    print()
    print("Whisper installation itself is working,")
    print("but transcription cannot be tested until a video is provided.")

    sys.exit(0)


print("✅ Video found:")
print(VIDEO_PATH)


# ============================================================
# CHECK VIDEO AUDIO
# ============================================================

print()
print("🔊 Checking audio track...")


audio_command = [
    "ffprobe",
    "-v",
    "error",
    "-select_streams",
    "a:0",
    "-show_entries",
    "stream=codec_name",
    "-of",
    "default=noprint_wrappers=1:nokey=1",
    VIDEO_PATH
]


try:

    audio_result = subprocess.run(
        audio_command,
        capture_output=True,
        text=True
    )

    audio_codec = audio_result.stdout.strip()

except Exception as e:

    print("❌ Could not check audio:")
    print(repr(e))

    sys.exit(1)


if not audio_codec:

    print("⚠️ NO AUDIO TRACK FOUND.")

    print()
    print("This video cannot be transcribed by Whisper.")

    print()
    print("Use a video containing speech/audio,")
    print("for example:")
    print("test_whisper.mp4")

    sys.exit(0)


print("✅ Audio found")
print("   Codec:", audio_codec)


# ============================================================
# LOAD WHISPER MODEL
# ============================================================

print()
print("=" * 60)
print("🧠 Loading Whisper AI")
print("=" * 60)

print()
print("Model: tiny")
print("This is being used for the first test.")
print("It is much lighter and faster than larger models.")


try:

    model = whisper.load_model(
        "tiny"
    )

    print()
    print("✅ Whisper model loaded successfully!")

except Exception as e:

    print()
    print("❌ Whisper model could not be loaded.")

    print()
    print("ERROR:")
    traceback.print_exc()

    sys.exit(1)


# ============================================================
# TRANSCRIBE VIDEO
# ============================================================

print()
print("=" * 60)
print("🎙️ STARTING WHISPER TRANSCRIPTION")
print("=" * 60)

print()
print("Video:")
print(VIDEO_PATH)

print()
print("⏳ Please wait...")


try:

    result = model.transcribe(
        VIDEO_PATH,
        language=None,
        fp16=False,
        verbose=False
    )

except Exception as e:

    print()
    print("❌ Whisper transcription failed.")

    print()
    print("ERROR:")
    traceback.print_exc()

    sys.exit(1)


# ============================================================
# GET TRANSCRIPT
# ============================================================

text = result.get(
    "text",
    ""
).strip()


print()
print("=" * 60)
print("📝 TRANSCRIPTION RESULT")
print("=" * 60)

if text:

    print()
    print(text)

else:

    print()
    print("⚠️ Whisper did not detect any speech.")


# ============================================================
# PRINT TIMESTAMPED SEGMENTS
# ============================================================

segments = result.get(
    "segments",
    []
)


print()
print("=" * 60)
print("⏱️ TIMESTAMPED SPEECH")
print("=" * 60)


if not segments:

    print()
    print("No speech segments detected.")

else:

    for index, segment in enumerate(
        segments,
        start=1
    ):

        start = float(
            segment.get("start", 0)
        )

        end = float(
            segment.get("end", 0)
        )

        segment_text = segment.get(
            "text",
            ""
        ).strip()

        print()
        print(
            f"{index:02d}. "
            f"{start:.2f}s → {end:.2f}s"
        )

        print(
            f"    {segment_text}"
        )


# ============================================================
# SUCCESS
# ============================================================

print()
print("=" * 60)
print("🎉 WHISPER TEST COMPLETED")
print("=" * 60)

print()
print("✅ PyTorch: Working")
print("✅ Whisper: Working")
print("✅ FFprobe: Working")
print("✅ Audio: Found")
print("✅ Speech analysis: Completed")

print()
print("VidioAI can now use Whisper for speech analysis.")

print()