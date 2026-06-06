
import subprocess
from huggingface_hub import snapshot_download, hf_hub_download

def sh(cmd):
    """Run a shell command, printing errors but not crashing the whole app."""
    try:
        subprocess.check_call(cmd, shell=True)
    except subprocess.CalledProcessError as e:
        print(f"⚠ Shell command failed (non-fatal): {cmd}\n  {e}")

# --- Install flash-attn (may take a while on first run) ---
sh("pip install flash-attn --no-build-isolation")
# --- Fix onnxruntime GPU ---
sh("pip uninstall onnxruntime onnxruntime-gpu -y")
sh("pip install onnxruntime-gpu")

import os
import shutil

# ==========================================================================
# HuggingFace Spaces compatibility: `spaces` module only exists on HF Spaces.
# On Lightning AI / local, we create a no-op fallback.
# ==========================================================================
try:
    import spaces
    _ON_HF_SPACES = True
except ImportError:
    _ON_HF_SPACES = False

    class _SpacesFallback:
        """No-op fallback for the HuggingFace `spaces` module."""
        @staticmethod
        def GPU(*args, **kwargs):
            """No-op decorator — on Lightning AI the GPU is always available."""
            def decorator(fn):
                return fn
            # Handle @spaces.GPU(duration=...) and @spaces.GPU
            if len(args) == 1 and callable(args[0]):
                return args[0]
            return decorator

    spaces = _SpacesFallback()
    print("ℹ Running outside HuggingFace Spaces — GPU decorator disabled.")

import io
import torch
import inspect
import pyannote.audio.core.task as task_module
from pathlib import Path
from pydub import AudioSegment
import math

import re
import tempfile
from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError
import subprocess

# Collect all classes from pyannote.audio.core.task
safe_globals = [torch.torch_version.TorchVersion]
for name, obj in inspect.getmembers(task_module):
    if inspect.isclass(obj):
        safe_globals.append(obj)

# Allow these classes to be used when unpickling weights with weights_only=True
torch.serialization.add_safe_globals(safe_globals)

from typing import List, Dict
import time
from time_util import timer
import os, pathlib, sys, ctypes
import uuid

# ==========================================================================
# Preload cuDNN CNN component — path varies by Python version / platform.
# ==========================================================================
_cudnn_paths = [
    "/usr/local/lib/python3.10/site-packages/nvidia/cudnn/lib/libcudnn_cnn.so.9",
    "/usr/local/lib/python3.12/site-packages/nvidia/cudnn/lib/libcudnn_cnn.so.9",
    "/usr/local/lib/python3.11/site-packages/nvidia/cudnn/lib/libcudnn_cnn.so.9",
]
_cudnn_loaded = False
for _p in _cudnn_paths:
    if os.path.exists(_p):
        try:
            ctypes.CDLL(_p)
            _cudnn_loaded = True
            print(f"✓ Loaded cuDNN from {_p}")
            break
        except OSError:
            pass
if not _cudnn_loaded:
    print("⚠ cuDNN CNN library not found at expected paths — may still work via LD_LIBRARY_PATH.")

# print(os.environ.get('LD_LIBRARY_PATH', ''))
import torch, ctranslate2, os

import numpy as np
from pydub import AudioSegment
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
import gradio as gr

from pydub import AudioSegment
import srt
import io
from pydub import AudioSegment
import math
from datetime import timedelta
import torchaudio
import tigersound.look2hear.models
from FastAudioSR import FASR
import librosa

file_path = hf_hub_download(repo_id="YatharthS/FlashSR", filename="upsampler.pth", local_dir=".")
upsampler = FASR(file_path)


current_dir = os.path.dirname(os.path.abspath(__file__))

# ==========================================================================
# ETHIOPIAN LANGUAGE MODIFICATION
# ==========================================================================

# SINGLE MULTILINGUAL MODEL
# Replace with your actual HuggingFace repository name
MULTILINGUAL_MODEL_REPO = "robadugna/rtts2"

# Download checkpoint at startup
TTS_CHECKPOINT_DIR = os.path.join(current_dir, "checkpoints_multilingual")
snapshot_download(MULTILINGUAL_MODEL_REPO, local_dir=TTS_CHECKPOINT_DIR)

# ---- Post-download fixups for config_amharic.yaml ----
import yaml
_cfg_path = os.path.join(TTS_CHECKPOINT_DIR, "config_amharic.yaml")
if os.path.exists(_cfg_path):
    with open(_cfg_path, "r") as f:
        _conf = yaml.safe_load(f)
    _conf_changed = False

    # Fix 1: Tokenizer path — config says "../tokenizers/am_om_ti_extended.model"
    # but the file is downloaded flat into TTS_CHECKPOINT_DIR.
    if "dataset" in _conf and "bpe_model" in _conf["dataset"]:
        _conf["dataset"]["bpe_model"] = os.path.join(TTS_CHECKPOINT_DIR, "am_om_ti_extended.model")
        _conf_changed = True

    # Fix 2: GPT checkpoint name — config says "gpt.pth" but the HF repo
    # only has "latest.pth". Rename the file so IndexTTS2 can find it.
    _gpt_expected = os.path.join(TTS_CHECKPOINT_DIR, "gpt.pth")
    _gpt_actual = os.path.join(TTS_CHECKPOINT_DIR, "latest.pth")
    if not os.path.exists(_gpt_expected) and os.path.exists(_gpt_actual):
        print("  ℹ Renaming latest.pth → gpt.pth (config expects gpt.pth)")
        os.rename(_gpt_actual, _gpt_expected)

    if _conf_changed:
        with open(_cfg_path, "w") as f:
            yaml.safe_dump(_conf, f)
        print("  ✓ config_amharic.yaml patched (tokenizer path fixed).")

# Copy checkpoints to torch hub cache (some models expect files there)
_torch_hub_dst = os.path.expanduser("~/.cache/torch/hub/checkpoints")
os.makedirs(_torch_hub_dst, exist_ok=True)
for _item in os.listdir(TTS_CHECKPOINT_DIR):
    _s = os.path.join(TTS_CHECKPOINT_DIR, _item)
    _d = os.path.join(_torch_hub_dst, _item)
    if os.path.isdir(_s):
        shutil.copytree(_s, _d, dirs_exist_ok=True)
    elif not os.path.exists(_d):
        shutil.copy2(_s, _d)
print("✓ Done copying checkpoints to torch hub cache!")

# ==========================================================================

dnr_model = tigersound.look2hear.models.TIGERDNR.from_pretrained("JusperLee/TIGER-DnR").to("cuda").eval()


from indextts.infer_v2 import IndexTTS2

MODE = 'local'

# LAZY-LOAD: Load the single multilingual TTS model on first use.
_multilingual_tts_model = None

def get_tts_model(lang_code: str) -> IndexTTS2:
    """Lazy-load and cache the single multilingual IndexTTS2 model."""
    global _multilingual_tts_model
    
    if _multilingual_tts_model is not None:
        return _multilingual_tts_model
    
    print("Loading multilingual IndexTTS2 model...")
    _multilingual_tts_model = IndexTTS2(
        model_dir=TTS_CHECKPOINT_DIR,
        cfg_path=os.path.join(TTS_CHECKPOINT_DIR, "config_amharic.yaml"),
        use_fp16=True,
        use_deepspeed=False,
        use_cuda_kernel=False,
    )
    print("  ✓ Multilingual model loaded!")
    return _multilingual_tts_model

# ==========================================================================
# FREE TRANSLATION API (replaces NLLB — saves ~7GB VRAM!)
# ==========================================================================
import requests as http_req
import json

TRANSLATION_API_URL = "https://dev-mapiz.pantheonsite.io/ymigxf/Api/"

# Human-readable target language names for the translation prompt
TARGET_LANG_NAMES = {
    "am":  "Amharic (አማርኛ)",
    "tir": "Tigrinya (ትግርኛ)",
    "om":  "Oromo (Afaan Oromoo)",
}

# Mapping from Gradio display names → internal language codes
LANG_DISPLAY_TO_CODE = {
    "Amharic (አማርኛ)":    "am",
    "Tigrinya (ትግርኛ)":   "tir",
    "Oromo (Afaan Oromoo)": "om",
}

def resolve_lang_code(target_lang: str) -> str:
    """Convert a display name or raw code to a valid internal language code."""
    if target_lang in LANG_DISPLAY_TO_CODE:
        return LANG_DISPLAY_TO_CODE[target_lang]
    if target_lang in TARGET_LANG_NAMES:
        return target_lang  # Already a valid code
    return "am"  # Fallback

# ==========================================================================

os.environ["PROCESSED_RESULTS"] = f"{os.getcwd()}/processed_results"

from lipsync import apply_lipsync
import logging

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)



def _safe_name(s: str) -> str:
    s = s.strip()
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", s)
    return s[:80] if s else "tiktok"



def download_tiktok_video(url: str) -> str:

    if not url or not url.strip():
        raise gr.Error("Please paste a TikTok link.")

    url = url.strip()

    out_dir = Path(tempfile.mkdtemp(prefix="tiktok_dl_"))
    outtmpl = str(out_dir / "%(title).80s_%(id)s.%(ext)s")

    ydl_opts = {
        "format": "bv*+ba/best",
        "merge_output_format": None,
        "postprocessors": [],
        "concurrent_fragment_downloads": 1,
        "ffmpeg_args": ["-threads", "2"],
        "outtmpl": outtmpl,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
    }

    try:
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)

            # yt-dlp can return a dict with requested_downloads or a direct filename
            # We'll try a few safe ways to get the final filepath.
            fp = None

            if isinstance(info, dict):
                # Most common: requested_downloads contains the output file
                req = info.get("requested_downloads") or []
                if req and isinstance(req, list) and isinstance(req[0], dict):
                    fp = req[0].get("filepath") or req[0].get("filename")

                # Another common: ydl.prepare_filename
                if not fp:
                    fp = ydl.prepare_filename(info)

            if not fp:
                # As a fallback, pick the newest file in our temp dir
                files = list(out_dir.glob("*"))
                if not files:
                    raise gr.Error("Download finished but no file was found.")
                fp = str(max(files, key=lambda p: p.stat().st_mtime))

            # Ensure Gradio can read it
            if not os.path.exists(fp):
                raise gr.Error("Downloaded file path does not exist.")


            return fp


    except DownloadError as e:
        msg = str(e)
        raise gr.Error(
            "Download failed. TikTok may be blocking requests.\n\n"
            "Try:\n"
            "• Using a different TikTok link\n"
            "• Providing a cookies.txt export (logged-in) in Advanced\n\n"
            f"Details: {msg[:800]}"
        )

def split_subtitles_max_duration(
    subtitles, 
    max_seconds: float = 10.0, 
    min_last_chunk_seconds: float = 1.0,
):
    """
    Take a list of srt.Subtitle and return a new list where
    no subtitle duration is longer than max_seconds, except that
    the *last* chunk is allowed to exceed max_seconds slightly
    if the leftover duration would otherwise be less than
    min_last_chunk_seconds.

    Text is split by words roughly evenly across the chunks.
    """
    max_td = timedelta(seconds=max_seconds)
    new_subs = []
    new_index = 1

    for sub in subtitles:
        start = sub.start
        end = sub.end
        duration = end - start
        total_secs = duration.total_seconds()

        # If already short enough, just copy it
        if total_secs <= max_seconds:
            new_subs.append(
                srt.Subtitle(
                    index=new_index,
                    start=start,
                    end=end,
                    content=sub.content,
                )
            )
            new_index += 1
            continue

        # Need to split this subtitle
        words = sub.content.split()
        if not words:
            # No text, skip
            continue

        # --- Determine number of chunks, avoiding tiny last chunk ---
        base_chunks = int(total_secs // max_seconds)
        remainder = total_secs - base_chunks * max_seconds

        if base_chunks == 0:
            # total_secs > max_seconds due to earlier check, but just in case
            num_chunks = 1
        else:
            if remainder == 0:
                num_chunks = base_chunks
            elif remainder < min_last_chunk_seconds:
                # Don't create a tiny last chunk; merge its time into previous chunks
                num_chunks = base_chunks
            else:
                num_chunks = base_chunks + 1

        # Ensure at least one chunk
        num_chunks = max(1, num_chunks)

        # Words per chunk (roughly even)
        words_per_chunk = max(1, int(math.ceil(len(words) / num_chunks)))

        chunk_start = start
        word_idx = 0

        for chunk_idx in range(num_chunks):
            # Last chunk takes us all the way to the original end,
            # so it can be slightly > max_seconds if needed.
            if chunk_idx == num_chunks - 1:
                chunk_end = end
            else:
                chunk_end = min(end, chunk_start + max_td)

            if chunk_end <= chunk_start:
                break

            chunk_words = words[word_idx:word_idx + words_per_chunk]
            word_idx += words_per_chunk

            if not chunk_words:
                break

            new_subs.append(
                srt.Subtitle(
                    index=new_index,
                    start=chunk_start,
                    end=chunk_end,
                    content=" ".join(chunk_words),
                )
            )
            new_index += 1

            chunk_start = chunk_end

    return new_subs

def split_text_into_chunks(text, max_chars=400):
    """
    Rough splitter: breaks text into chunks <= max_chars, 
    preferring to split at sentence boundaries, then spaces.
    Supports both Latin and Ethiopic (Ge'ez) punctuation.
    """
    text = text.strip()
    chunks = []

    while len(text) > max_chars:
        # Try to split at the last sentence end before max_chars
        # Include Ethiopic punctuation: ። (full stop), ፧ (question), ፣ (comma)
        split_at = max(
            text.rfind(". ", 0, max_chars),
            text.rfind("! ", 0, max_chars),
            text.rfind("? ", 0, max_chars),
            text.rfind("\u1362 ", 0, max_chars),   # ። Ethiopic full stop
            text.rfind("\u1367 ", 0, max_chars),   # ፧ Ethiopic question mark
            text.rfind("\u1363 ", 0, max_chars),   # ፣ Ethiopic comma
            text.rfind("\u1364 ", 0, max_chars),   # ፤ Ethiopic semicolon
        )

        # If there was no sentence boundary, fall back to last space
        if split_at == -1:
            split_at = text.rfind(" ", 0, max_chars)

        # If still nothing, just hard cut
        if split_at == -1:
            split_at = max_chars

        chunk = text[:split_at + 1].strip()
        chunks.append(chunk)
        text = text[split_at + 1 :].strip()

    if text:
        chunks.append(text)

    return chunks

def sh(cmd): subprocess.check_call(cmd, shell=True)
    
# sh("find / -name \"libcudnn*\" 2>/dev/null")
# --------------------
# CONFIG
# --------------------
# ETHIOPIAN MODIFICATION: Use large-v3 for much better Ethiopian language recognition
MODEL_SIZE = "large-v3"            # CHANGED from "medium" — critical for Am/Tir/Om
MIN_SEGMENT_SECONDS = 0.5        # only transcribe segments longer than this

# If your pyannote pipeline needs a HF token, set it here or via env var:
# HUGGINGFACE_TOKEN = "hf_..."
HF_TOKEN = os.getenv("HF_TOKEN", None)

# --------------------
# LOAD GLOBAL MODELS (ONCE)
# --------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading pyannote diarization model...")
diarization_pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1"
)

# --------------------
# HELPERS
# --------------------
def format_timestamp(ts: float) -> str:
    """Convert seconds to SRT timestamp format."""
    hrs = int(ts // 3600)
    mins = int((ts % 3600) // 60)
    secs = int(ts % 60)
    ms = int((ts - int(ts)) * 1000)
    return f"{hrs:02d}:{mins:02d}:{secs:02d},{ms:03d}"


def extract_audio_to_wav(input_video: str, output_dir: str):

    audio_file = os.path.join(output_dir, "audio_og.wav")
    background_file = os.path.join(output_dir, "background_og.wav")
    vocal_file = os.path.join(output_dir, "vocal_og.wav")
    effect_file = os.path.join(output_dir, "effect_og.wav")

    audio_16k_file = os.path.join(output_dir, "audio_16k.wav")
    
    video_path = input_video
    separator_dir = Path(os.path.join(output_dir, "separator_directory"))
    os.makedirs(separator_dir, exist_ok=True)


    # Extract raw audio
    cmd = [
        "ffmpeg",
        "-loglevel", "error",
        "-y",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "44100",
        "-ac", "2",
        audio_file
    ]
    subprocess.run(cmd, check=True)

    audio, sr = torchaudio.load(audio_file)
    audio = audio.to("cuda")
    
    with torch.no_grad():
        dialog, effect, music = dnr_model(audio[None])
    
    torchaudio.save(vocal_file, dialog.cpu(), sr)
    torchaudio.save(effect_file, effect.cpu(), sr)
    torchaudio.save(background_file, music.cpu(), sr)

    # Convert vocals to 16k mono
    cmd = [
        "ffmpeg",
        "-loglevel", "error",
        "-y",
        "-i", vocal_file,
        "-ac", "1",
        "-ar", "16000",
        "-acodec", "pcm_s16le",
        audio_16k_file
    ]
    subprocess.run(cmd, check=True)
    
    return audio_file, effect_file, background_file, audio_16k_file, vocal_file

def diarize_audio(audio_path: str) -> List[Dict]:
    """Run pyannote diarization and return segments."""

    diarization_pipeline.to(torch.device(device))

    with ProgressHook() as hook:
        diarization_result = diarization_pipeline(audio_path, hook=hook)

    segments = []
    for segment, _, speaker in diarization_result.itertracks(yield_label=True):
        duration = segment.end - segment.start
        if duration >= MIN_SEGMENT_SECONDS:          
            segments.append(
                {
                    "start": float(segment.start),
                    "end": float(segment.end),
                    "speaker": speaker,
                }
            )

    segments.sort(key=lambda x: x["start"])
    return segments

def chunk_to_float32(chunk: AudioSegment) -> np.ndarray:
    """Convert a pydub chunk to mono 16kHz float32 numpy array in [-1, 1]."""
    chunk = chunk.set_frame_rate(16000).set_channels(1)
    samples = np.array(chunk.get_array_of_samples())

    # Normalize based on sample width
    if chunk.sample_width == 2:  # 16-bit
        samples = samples.astype(np.float32) / 32768.0
    elif chunk.sample_width == 4:  # 32-bit
        samples = samples.astype(np.float32) / 2147483648.0
    else:
        samples = samples.astype(np.float32)

    return samples

# ==========================================================================
# ETHIOPIAN MODIFICATION: Translation Functions
# ==========================================================================

def detect_whisper_language(whisper_model, samples: np.ndarray) -> str:
    """
    Detect the language of the audio segment using Whisper.
    Returns the ISO 639-1 language code (e.g., 'en', 'fr', 'es').
    """
    segments, info = whisper_model.transcribe(
        samples,
        beam_size=1,
        vad_filter=False,
    )
    return info.language

def translate_text_api(text: str, source_lang: str, target_lang_code: str) -> str:
    """
    Translate text to the target Ethiopian language using the free Mapiz API.
    Uses OpenAI-compatible chat completions format.

    Args:
        text: The text to translate
        source_lang: Whisper-detected language code (e.g., 'en', 'fr', 'es')
        target_lang_code: One of 'am', 'tir', 'om'

    Returns:
        Translated text in the target Ethiopian script
    """
    if not text.strip():
        return text

    target_name = TARGET_LANG_NAMES.get(target_lang_code, "Amharic")
    
    system_prompt = (
        f"You are a professional translator. "
        f"Translate the following text from {source_lang} to {target_name}. "
        f"Output ONLY the translated text, nothing else. "
        f"Do not add quotes, explanations, or any extra text."
    )

    payload = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
        "stream": False,
        "temperature": 0.3,
    }

    headers = {
        "Content-Type": "application/json",
    }

    last_error = None
    for attempt in range(3):
        try:
            resp = http_req.post(
                TRANSLATION_API_URL,
                headers=headers,
                json=payload,
                timeout=30,
            )

            if resp.status_code == 200:
                data = resp.json()
                
                # --- OpenAI-compatible response format ---
                # {"choices": [{"message": {"content": "translated text"}}]}
                if "choices" in data and len(data["choices"]) > 0:
                    translated = data["choices"][0]["message"]["content"].strip()
                    if translated:
                        return translated

                # --- Alternative: {"response": "translated text"} ---
                if "response" in data:
                    translated = data["response"].strip()
                    if translated:
                        return translated

                # --- Alternative: {"text": "translated text"} ---
                if "text" in data:
                    translated = data["text"].strip()
                    if translated:
                        return translated

                # --- Alternative: plain string in data ---
                if isinstance(data, str) and data.strip():
                    return data.strip()

                last_error = f"API returned 200 but unexpected format: {str(data)[:200]}"
                print(f"  ⚠ {last_error}")
                
            else:
                last_error = f"API returned HTTP {resp.status_code}: {resp.text[:200]}"
                print(f"  ⚠ {last_error}")

        except http_req.exceptions.Timeout:
            last_error = f"API timeout (attempt {attempt + 1}/3)"
            print(f"  ⚠ {last_error}")
            time.sleep(1)
        except Exception as e:
            last_error = f"API error (attempt {attempt + 1}/3): {e}"
            print(f"  ⚠ {last_error}")
            time.sleep(1)

    # CRITICAL: Do NOT silently return English text — it will produce garbled TTS output.
    # Instead, mark the failure clearly so downstream code can handle it.
    print(f"  ✗ Translation FAILED for: {text[:80]} | Last error: {last_error}")
    return f"[UNTRANSLATED] {text}"


def test_translation_api():
    """
    DEBUG: Call this function standalone to test if the API works.
    Usage: python modified_app.py --test-api
    """
    print("=" * 60)
    print("TESTING TRANSLATION API")
    print(f"Endpoint: {TRANSLATION_API_URL}")
    print("=" * 60)

    test_cases = [
        ("Hello, how are you today?", "en", "am"),
        ("The weather is beautiful.", "en", "am"),
        ("Good morning everyone!", "en", "tir"),
        ("I love this place.", "en", "om"),
    ]

    for text, src, tgt in test_cases:
        print(f"\n  [{src} → {tgt}] \"{text}\"")
        result = translate_text_api(text, src, tgt)
        print(f"  → \"{result}\"")

    print("\n" + "=" * 60)

# NOTE: transcribe_segment_words_ethiopian was removed — its logic is now
# consolidated into transcribe_segment_words() below.

def transcribe_segment(whisper_model, samples: np.ndarray) -> str:
    """Transcribe a single segment with faster-whisper (ASR only, no translation)."""
    segment_text_parts = []

    segments, info = whisper_model.transcribe(
        samples,
        beam_size=1,
        vad_filter=False,
        condition_on_previous_text=True,
        task="transcribe",                # CHANGED from "translate"
        word_timestamps=True,
    )

    for seg in segments:
        if seg.text:
            segment_text_parts.append(seg.text.strip())

    return " ".join(segment_text_parts)

def transcribe_segment_words(
    whisper_model,
    samples: np.ndarray,
    offset_sec: float,
    speaker: str | None = None,
    target_lang: str = "am",
):
    """
    Two-step transcription for Ethiopian dubbing:
    1. Whisper ASR — transcribes the source audio (auto-detects source language)
    2. Translation API — translates each segment to the target Ethiopian language
    
    Returns a list of word dicts with absolute timestamps.
    Each word carries both the original text (for timing) and the segment-level
    translated text (for subtitle/TTS content).
    """
    words_out = []

    # Step 1: Whisper ASR — always auto-detect source language.
    # NOTE: Do NOT pass language=target_lang here! The SOURCE audio is not
    # in the target language. Also, Whisper does not support 'tir' (Tigrinya)
    # as a language code, so passing it would crash.
    segments, info = whisper_model.transcribe(
        samples,
        beam_size=1,
        vad_filter=False,
        condition_on_previous_text=False,
        task="transcribe",                # ASR only, no Whisper translation
        word_timestamps=True,
        language=None,                     # Auto-detect source language
    )
    
    source_lang = info.language
    print(f"    Source language detected: {source_lang}")

    for seg in segments:
        if not seg.words:
            continue
        
        seg_text = seg.text.strip()
        if not seg_text:
            continue
        
        # Step 2: Translate the entire segment text via API
        translated = translate_text_api(seg_text, source_lang, target_lang)
        
        # Skip segments where translation completely failed
        if translated.startswith("[UNTRANSLATED]"):
            print(f"    ⚠ Skipping untranslated segment: {seg_text[:50]}")
            # Still include words with original text so timing is preserved
            translated = seg_text  # Fallback to source text
        
        print(f"    '{seg_text[:50]}' → '{translated[:50]}'")
        
        # Store the segment-level translation with each word for timing alignment
        # The 'seg_id' groups words that share the same translation
        seg_id = id(seg)
        for w in seg.words:
            words_out.append(
                {
                    "start": offset_sec + float(w.start),
                    "end": offset_sec + float(w.end),
                    "text": w.word,
                    "translated": translated,
                    "seg_id": seg_id,       # Groups words from same Whisper segment
                    "speaker": speaker,
                }
            )

    return words_out

# ==========================================================================
# ETHIOPIAN MODIFICATION: Subtitle grouping now uses translated text
# ==========================================================================

def words_to_subtitles(words, max_seconds: float = 10.0, target_lang: str = "am"):
    """
    Group word-level timings into SRT subtitles, each up to max_seconds long.
    Uses the 'translated' field for subtitle content (text in Amharic/Tigrinya/Oromo).
    
    Words from the same Whisper segment share the same 'seg_id' and 'translated' text.
    When a subtitle spans multiple segments, we concatenate the unique translations
    in order (not just take the last one, which would lose earlier segments).
    """
    words = sorted(words, key=lambda w: w["start"])

    subtitles = []
    current_words = []
    current_start = None
    current_speaker = None

    index = 1

    def _build_subtitle_text(word_list):
        """Build subtitle text from grouped words, deduplicating by segment."""
        seen_seg_ids = set()
        parts = []
        for w in word_list:
            seg_id = w.get("seg_id")
            translated = w.get("translated", w.get("text", ""))
            if seg_id is not None:
                if seg_id not in seen_seg_ids:
                    seen_seg_ids.add(seg_id)
                    parts.append(translated.strip())
            else:
                # Fallback for words without seg_id
                parts.append(translated.strip())
        return " ".join(parts) if parts else ""

    for w in words:
        w_start = w["start"]
        w_end = w["end"]
        w_speaker = w.get("speaker")

        if current_start is None:
            current_start = w_start
            current_words = [w]
            current_speaker = w_speaker
            continue

        speaker_changed = (w_speaker != current_speaker)
        duration_if_added = w_end - current_start
        exceeds_max = duration_if_added > max_seconds

        if (speaker_changed or exceeds_max) and current_words:
            text = _build_subtitle_text(current_words)
            
            sub_start = current_start
            sub_end = current_words[-1]["end"]

            if text:  # Only create subtitle if there's actual text
                subtitles.append(
                    srt.Subtitle(
                        index=index,
                        start=timedelta(seconds=sub_start),
                        end=timedelta(seconds=sub_end),
                        content=text,
                    )
                )
                index += 1

            current_start = w_start
            current_words = [w]
            current_speaker = w_speaker
        else:
            current_words.append(w)

    # flush last subtitle
    if current_words:
        text = _build_subtitle_text(current_words)
        sub_start = current_start
        sub_end = current_words[-1]["end"]
        if text:
            subtitles.append(
                srt.Subtitle(
                    index=index,
                    start=timedelta(seconds=sub_start),
                    end=timedelta(seconds=sub_end),
                    content=text,
                )
            )

    return subtitles

# ==========================================================================

def build_srt(segments: List[Dict], audio_wav: str, out_srt_path: str, target_lang: str = "am"):
    """
    Generate SRT file from diarized segments and audio.
    ETHIOPIAN MODIFICATION: Now takes target_lang and uses two-step translation.
    """
    audio = AudioSegment.from_file(audio_wav)

    print(f"Loading faster-whisper model ({MODEL_SIZE})...")
    whisper_model = WhisperModel(
        MODEL_SIZE,
        device="cuda",
        compute_type="float16",
    )

    all_words = []

    for i, seg in enumerate(segments, start=1):
        start_sec = seg["start"]
        end_sec = seg["end"]
        speaker = seg["speaker"]

        start_ms = int(start_sec * 1000)
        end_ms = int(end_sec * 1000)
        chunk = audio[start_ms:end_ms]

        samples = chunk_to_float32(chunk)

        # ETHIOPIAN MODIFICATION: Use target_lang-aware transcription
        seg_words = transcribe_segment_words(
            whisper_model,
            samples,
            offset_sec=start_sec,
            speaker=speaker,
            target_lang=target_lang,
        )

        all_words.extend(seg_words)
        print(f"Diar segment {i} ({speaker}): {len(seg_words)} words")

    # group words into ≤10s subtitles
    subtitles = words_to_subtitles(all_words, max_seconds=10.0, target_lang=target_lang)

    # write SRT
    with open(out_srt_path, "w", encoding="utf-8") as f:
        f.write(srt.compose(subtitles))

def resolve_video_input(x: str):
    """
    Accepts:
      - TikTok URL (http/https) -> downloads and returns filepath
      - Local filepath -> returns it (if exists)
      - Gradio dict {"name"/"path"} -> extracts it
    """
    if x is None:
        raise gr.Error("Please provide a TikTok URL or a video file/path.")

    # If gradio gives a dict (depends on component/version)
    if isinstance(x, dict):
        x = x.get("name") or x.get("path")

    if not isinstance(x, str):
        raise gr.Error("Invalid input. Provide a TikTok URL or a video file/path.")

    x = x.strip()
    if not x:
        raise gr.Error("Please provide a TikTok URL or a video file/path.")

    # URL -> download
    if re.match(r"^https?://", x, re.IGNORECASE):
        return download_tiktok_video(x)

    # Otherwise treat as local path
    if not os.path.exists(x):
        raise gr.Error("Video path does not exist. Provide a valid local path or a TikTok URL.")

    return x

# ==========================================================================
# ETHIOPIAN MODIFICATION: Process functions now take target_lang
# ==========================================================================

def translate_video(video_file, url_or_path, duration, target_lang="am", session_id=None, progress=gr.Progress(track_tqdm=True)):

    if video_file == None:
        url_or_path = url_or_path
    else:
        url_or_path = video_file
    video_file = resolve_video_input(url_or_path)
    return process_video(video_file, False, duration, target_lang, session_id, progress)


def translate_lipsync_video(video_file, url_or_path, duration, target_lang="am", session_id=None, progress=gr.Progress(track_tqdm=True)):

    if video_file == None:
        url_or_path = url_or_path
    else:
        url_or_path = video_file
        
    video_file = resolve_video_input(url_or_path)
    return process_video(video_file, True, duration, target_lang, session_id, progress)


def run_example(video_file, allow_lipsync, duration, target_lang="am", session_id = None, progress=gr.Progress(track_tqdm=True)):

    with timer("processed"):
        result = process_video(video_file, allow_lipsync, duration, target_lang, session_id, progress)

    return result

def get_duration(video_file, allow_lipsync, duration, session_id, progress):

    if allow_lipsync:
        return (60 + 30 * (duration) // 5) // 2
    else:
        return (60 + 20 * (duration) // 30) // 2
        
@spaces.GPU(duration=get_duration, size='xlarge')
def process_video(video_file, allow_lipsync, duration, target_lang="am", session_id = None, progress=gr.Progress(track_tqdm=True)):
    """
    Main processing pipeline for Ethiopian dubbing.
    Takes target_lang (display name or code) and uses the corresponding
    fine-tuned IndexTTS2 model for that language.
    """
    import onnxruntime as ort

    # CRITICAL: Convert display name "Amharic (አማርኛ)" -> "am" etc.
    target_lang = resolve_lang_code(target_lang)
    print(f"Target language resolved to: {target_lang}")

    if session_id == None:
        session_id = uuid.uuid4().hex

    output_dir = os.path.join(os.environ["PROCESSED_RESULTS"], session_id)
    os.makedirs(output_dir, exist_ok=True)

    # Get the TTS model for the target language (lazy-loaded)
    tts = get_tts_model(target_lang)

    # Gradio's File/Video component gives dict or str depending on version
    if isinstance(video_file, dict):
        video_path = video_file.get("name") or video_file.get("path")
    else:
        video_path = video_file

    if video_path is None or not os.path.exists(video_path):
        raise gr.Error("Could not read uploaded video file.")

    # Create temp directory to hold WAV + SRT
    srt_path = os.path.join(output_dir, "diarized_translated.srt")

    src_video_path = video_path

    cropped_video_path = os.path.join(output_dir, "input_30s.mp4")

    duration_s = int(duration)

    print(f"duration_s:{duration_s}")
    
    cmd = [                                                              
        "ffmpeg",                                                        
        "-y",                                                            
        "-i", src_video_path,                                            
        "-t", f"{duration_s}",                              
        "-c", "copy",          # stream copy, no re-encode               
        cropped_video_path,                                              
    ]                                                                    
    subprocess.run(cmd, check=True)                                      
    video_path = cropped_video_path                                       

    # 1. Extract audio
    audio_wav, effect_wav, background_wav, audio_16k_wav, vocal_wav = extract_audio_to_wav(video_path, output_dir)

    # 2. Diarization
    segments = diarize_audio(audio_16k_wav)
    if not segments:
        raise gr.Error("No valid speech segments found for diarization.")

    # 3. Build SRT — ETHIOPIAN MODIFICATION: Now uses two-step translation
    with timer("Generating srt"):
        build_srt(segments, audio_16k_wav, srt_path, target_lang=target_lang)

    # ---- ORIGINAL SRT (used for TTS) ----
    with open(srt_path, "r", encoding="utf-8") as f:
        srt_data = f.read()

    subtitles = list(srt.parse(srt_data))

    tts_subtitles = subtitles

    max10_subtitles = tts_subtitles

    tts_subtitles = max10_subtitles
    
    srt_10s_path = os.path.join(output_dir, "diarized_translated_max10s.srt")
    with open(srt_10s_path, "w", encoding="utf-8") as f:
        f.write(srt.compose(max10_subtitles))

    # ---- TTS USING ORIGINAL SRT ----
    last_end_seconds = tts_subtitles[-1].end.total_seconds()
    total_ms = int((last_end_seconds + 1) * 1000)

    timeline = AudioSegment.silent(duration=total_ms)

    original_audio = AudioSegment.from_file(audio_wav)

    MAX_BATCH_MS = 300_000  # ~5 minutes of target subtitle duration per batch

    with timer("Generating speech"):
        num_subs = len(tts_subtitles)
        idx = 0

        while idx < num_subs:
            spk_prompts = []
            texts = []
            out_paths = []
            starts_ms = []
            target_ms_list = []
            batch_ms_sum = 0

            batch_start = idx

            while idx < num_subs:
                sub = tts_subtitles[idx]

                start_ms = int(sub.start.total_seconds() * 1000)
                end_ms = int(sub.end.total_seconds() * 1000)
                target_ms = max(end_ms - start_ms, 0)

                if batch_ms_sum + target_ms > MAX_BATCH_MS and len(target_ms_list) > 0:
                    break

                global_idx = idx

                src_chunk = original_audio[start_ms:end_ms]
                src_prompt_path = os.path.join(output_dir, f"src_prompt_{global_idx}.wav")
                src_chunk.export(src_prompt_path, format="wav")

                text = sub.content.replace("\n", " ")
                out_path = os.path.join(output_dir, f"gen_{global_idx}.wav")

                spk_prompts.append(src_prompt_path)
                texts.append(text)
                out_paths.append(out_path)
                starts_ms.append(start_ms)
                target_ms_list.append(target_ms)

                batch_ms_sum += target_ms
                idx += 1

            print(f"batch from {batch_start} to {idx - 1}, batch_ms_sum: {batch_ms_sum}")

            do_sample = True
            top_p = 0.8
            top_k = 30
            temperature = 0.8
            length_penalty = 0.0
            num_beams = 3
            # Lowered from 10.0: Ethiopian languages have natural repetitions
            # (e.g., Amharic emphasis patterns like "ጥሩ ጥሩ"). 10.0 suppresses these.
            repetition_penalty = 3.0
            max_mel_tokens = 1500

            tts_outputs = tts.infer_batch(
                spk_audio_prompts=spk_prompts,
                texts=texts,
                output_paths=out_paths,
                emo_audio_prompts=None,
                emo_alpha=1.0,
                emo_vectors=None,
                use_emo_text=False,
                emo_texts=None,
                use_random=False,
                interval_silence=200,
                verbose=False,
                # Increased from 120: Ge'ez (Amharic/Tigrinya) characters are syllabic
                # and byte-level tokenizers may expand each char to 3-4 tokens.
                # 120 tokens is too few for meaningful Ethiopic sentences.
                # Oromo (Latin script) is fine at 120 but benefits from headroom too.
                max_text_tokens_per_segment=250,
                speed=1.0,
                target_length_ms=target_ms_list,
                do_sample=do_sample,
                top_p=top_p,
                top_k=top_k,
                temperature=temperature,
                length_penalty=length_penalty,
                num_beams=num_beams,
                repetition_penalty=repetition_penalty,
                max_mel_tokens=max_mel_tokens,
            )

            for local_idx, out_path in enumerate(tts_outputs):
                start_ms = starts_ms[local_idx]

                seg = AudioSegment.from_file(out_path, format="wav")
                seg = seg - 2
                timeline = timeline.overlay(seg, position=start_ms)

                os.remove(out_path)
                os.remove(spk_prompts[local_idx])

    # Bring back original dialog in the gaps
    dialog = AudioSegment.from_file(vocal_wav)
    dialog = dialog.set_frame_rate(timeline.frame_rate).set_channels(timeline.channels)

    total_len_ms = len(timeline)

    speech_regions = []
    for sub in tts_subtitles:
        start_ms = int(sub.start.total_seconds() * 1000)
        end_ms = int(sub.end.total_seconds() * 1000)
        start_ms = max(0, min(start_ms, total_len_ms))
        end_ms = max(0, min(end_ms, total_len_ms))
        if end_ms > start_ms:
            speech_regions.append((start_ms, end_ms))

    speech_regions.sort()
    merged = []
    for s, e in speech_regions:
        if not merged:
            merged.append([s, e])
        else:
            last_s, last_e = merged[-1]
            if s <= last_e:
                merged[-1][1] = max(last_e, e)
            else:
                merged.append([s, e])

    gaps = []
    cursor = 0
    for s, e in merged:
        if cursor < s:
            gaps.append((cursor, s))
        cursor = max(cursor, e)
    if cursor < total_len_ms:
        gaps.append((cursor, total_len_ms))

    MIN_GAP_MS = 10

    for g_start, g_end in gaps:
        if g_end - g_start < MIN_GAP_MS:
            continue

        original_chunk = dialog[g_start:g_end]
        original_chunk = original_chunk + 6

        timeline = timeline.overlay(original_chunk, position=g_start)


    video_in = video_path
    audio_in = output_dir + "/final_output.wav"
    audio_16k_in = output_dir + "/final_16k_output.wav"
    
    # Mix background + new TTS vocal
    
    if background_wav is not None:
        eff = AudioSegment.from_file(effect_wav)
        bg = AudioSegment.from_file(background_wav)

        if len(eff) < len(timeline):
            loops = math.ceil(len(timeline) / len(eff))
            eff = eff * loops
                    
        if len(bg) < len(timeline):
            loops = math.ceil(len(timeline) / len(bg))
            bg = bg * loops

        eff = eff[:len(timeline)]
        bg = bg[:len(timeline)]
    
        bg = bg + 2
        eff = eff + 2
    
        eff_timeline = eff.overlay(timeline)
        final_audio = bg.overlay(eff_timeline)
        final_16k_audio = timeline.set_frame_rate(16000).set_channels(1)
    else:
        final_audio = timeline
        final_16k_audio = timeline
    
    final_audio.export(audio_in, format="wav")
    final_16k_audio.export(audio_16k_in, format="wav")
   
    print(f"Done! Saved to {audio_in}")

    lipsynced_video = output_dir + "/output_with_lipsync_16k.mp4"

    if allow_lipsync:
        apply_lipsync(video_in, audio_16k_in, lipsynced_video)
    else:
        lipsynced_video = video_in

    video_out = output_dir + "/output_with_lipsync.mp4"

    audio_in_upsampled = output_dir + "/final_output_upsampled.wav"

    y, _ = librosa.load(audio_in, sr=16000)
    lowres_wav = torch.from_numpy(y).unsqueeze(0)
    
    new_wav = upsampler.run(lowres_wav)
    
    new_wav = torch.as_tensor(new_wav)
    if new_wav.dim() == 1:
        new_wav = new_wav.unsqueeze(0)
    elif new_wav.dim() == 2 and new_wav.size(0) != 1:
        new_wav = new_wav[0].unsqueeze(0)
    elif new_wav.dim() == 3:
        new_wav = new_wav[0]
    
    new_wav = new_wav.contiguous().to(torch.float32).cpu()
    new_wav = torch.clamp(new_wav, -1.0, 1.0)
    
    torchaudio.save(audio_in_upsampled, new_wav, 48000)

    
    cmd = [
        "ffmpeg",
        "-loglevel", "error",
        "-y",
        "-i", lipsynced_video,
        "-i", audio_in_upsampled,
        "-c:v", "copy",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-shortest",
        video_out,
    ]


    subprocess.run(cmd, check=True)


    return video_out, srt_10s_path, audio_16k_in



css = """
    #col-container {
        margin: 0 auto;
        max-width: 1600px;
    }
    #modal-container {
    width: 100vw;
    height: 100vh;
    display: flex;
    justify-content: center;
    align-items: center;
    }
    #modal-content {
    width: 100%;
    max-width: 700px;
    margin: 0 auto;
    border-radius: 8px;
    padding: 1.5rem;
    }
    #step-column {
        padding: 10px;
        border-radius: 8px;
        box-shadow: var(--card-shadow);
        margin: 10px;
    }
    #col-showcase {
        margin: 0 auto;
        max-width: 1100px;
    }
    .button-gradient {
        background: linear-gradient(45deg, rgb(255, 65, 108), rgb(255, 75, 43), rgb(255, 155, 0), rgb(255, 65, 108)) 0% 0% / 400% 400%;
        border: none;
        padding: 14px 28px;
        font-size: 16px;
        font-weight: bold;
        color: white;
        border-radius: 10px;
        cursor: pointer;
        transition: 0.3s ease-in-out;
        animation: 2s linear 0s infinite normal none running gradientAnimation;
        box-shadow: rgba(255, 65, 108, 0.6) 0px 4px 10px;
    }
    .toggle-container {
    display: inline-flex;
    background-color: #ffd6ff;
    border-radius: 9999px;
    padding: 4px;
    position: relative;
    width: fit-content;
    font-family: sans-serif;
    }
    .toggle-container input[type="radio"] {
    display: none;
    }
    .toggle-container label {
    position: relative;
    z-index: 2;
    flex: 1;
    text-align: center;
    font-weight: 700;
    color: #4b2ab5;
    padding: 6px 22px;
    border-radius: 9999px;
    cursor: pointer;
    transition: color 0.25s ease;
    }
    .toggle-highlight {
    position: absolute;
    top: 4px;
    left: 4px;
    width: calc(50% - 4px);
    height: calc(100% - 8px);
    background-color: #4b2ab5;
    border-radius: 9999px;
    transition: transform 0.25s ease;
    z-index: 1;
    }
    #true:checked ~ label[for="true"] {
    color: #ffd6ff;
    }
    #false:checked ~ label[for="false"] {
    color: #ffd6ff;
    }
    #false:checked ~ .toggle-highlight {
    transform: translateX(100%);
    }
    """

def _fmt_seconds(sec: int) -> str:
    sec = int(sec)
    return f"{sec}s"

def compute_etas(duration_value: int):
    t_no = get_duration(None, False, int(duration_value), None, None)
    t_ls = get_duration(None, True,  int(duration_value), None, None)

    md_no = f"**Estimated time (Translate):** `{_fmt_seconds(t_no)}`"
    md_ls = f"**Estimated time (Translate + Lipsync):** `{_fmt_seconds(t_ls)}`"
    return md_no, md_ls
    
def cleanup(request: gr.Request):

    sid = request.session_hash
    if sid:
        print(f"{sid} left")
        d1 = os.path.join(os.environ["PROCESSED_RESULTS"], sid)
        shutil.rmtree(d1, ignore_errors=True)
        
def start_session(request: gr.Request):

    return request.session_hash

# ==========================================================================
# ETHIOPIAN MODIFICATION: Gradio UI with Language Selector
# ==========================================================================

# Note: LANG_DISPLAY_TO_CODE is defined near the top of the file.

with gr.Blocks(css=css) as demo:

    session_state = gr.State()
    demo.load(start_session, outputs=[session_state])

    with gr.Column(elem_id="col-container"):
        gr.HTML(
            """
            <div style="text-align: center;">
                <p style="font-size:16px; display: inline; margin: 0;">
                    Translate and lipsync your clips from any language to
                    <strong>🇪🇹 Amharic / Tigrinya / Oromo</strong>
                </p>
            </div>
            <div style="text-align: center;">
                <p style="font-size:16px; display: inline; margin: 0;">
                    <strong>OutofLipSync-🇪🇹</strong>
                </p>
                <p style="font-size:16px; display: inline; margin: 0;">
                    -- Based on OutofLipSync by
                </p>
                <a href="https://huggingface.co/alexnasa" style="display: inline-block; vertical-align: middle; margin-left: 0.5em;">
                    <img src="https://img.shields.io/badge/🤗-alexnasa-yellow.svg">
                </a>
            </div>
            """
        )

        with gr.Row():
            with gr.Column(elem_id="step-column"):

                with gr.Tab("Video"):

                     with gr.Column():

                        gr.HTML("""
                            <div>
                                <span style="font-size: 24px;">1. Upload a Video</span><br>
                            </div>
                            """)
                        
                        video_input = gr.Video(
                            label="OG Clip",
                            height=512
                        )
                         
                with gr.Tab("TikTok URL"):

                    with gr.Column():

                        gr.HTML("""
                            <div>
                                <span style="font-size: 24px;">1. Paste TikTok link Here</span><br>
                            </div>
                            """)
                        
                        url_in = gr.Textbox(label="TikTok URL", placeholder="https://www.tiktok.com/@user/video/...")

    # ETHIOPIAN MODIFICATION: Language selector dropdown
                target_lang_dropdown = gr.Dropdown(
                    choices=list(LANG_DISPLAY_TO_CODE.keys()),
                    value="Amharic (አማርኛ)",
                    label="🇪🇹 Target Language",
                    info="Choose which Ethiopian language to dub into"
                )

                duration = gr.Slider(5, 120, 10, step=1, label="Duration(s)")

            with gr.Column(elem_id="step-column"):
                gr.HTML("""
                <div>
                    <span style="font-size: 24px;">2. Translate to 🇪🇹 + 💋 </span><br>
                </div>
                """)

                video_output = gr.Video(label="Output", height=512)
                lipsync = gr.Checkbox(label="Lipsync", value=False, visible=False)

                eta_translate_md = gr.Markdown("")
                eta_lipsync_md = gr.Markdown("")

                translate_btn = gr.Button("🇪🇹 Translate to Ethiopian")
                translate_lipsync_btn = gr.Button("🇪🇹 Translate + 💋 Lipsync", variant='primary', elem_classes="button-gradient")
        
            with gr.Column(elem_id="step-column"):
                gr.HTML("""
                <div>
                    <span style="font-size: 24px;">Outputs </span><br>
                </div>
                """)
                vocal_16k_output = gr.File(label="Vocal 16k", visible=False)
                srt_output = gr.File(label="Download translated Ethiopian SRT", visible=True)

    translate_btn.click(
        fn=translate_video,
        inputs=[video_input, url_in, duration, target_lang_dropdown, session_state],
        outputs=[video_output, srt_output, vocal_16k_output],
    )
    
    translate_lipsync_btn.click(
        fn=translate_lipsync_video,
        inputs=[video_input, url_in, duration, target_lang_dropdown, session_state],
        outputs=[video_output, srt_output, vocal_16k_output],
    )

    duration.change(
        fn=compute_etas,
        inputs=[duration],
        outputs=[eta_translate_md, eta_lipsync_md],
    )



if __name__ == "__main__":
    import sys
    if "--test-api" in sys.argv:
        test_translation_api()
    else:
        demo.unload(cleanup)
        demo.queue()
        demo.launch(ssr_mode=False)
