
import subprocess
from huggingface_hub import snapshot_download, hf_hub_download

def sh(cmd): subprocess.check_call(cmd, shell=True)

sh("pip uninstall onnxruntime onnxruntime-gpu -y && pip install onnxruntime-gpu")

import os
import shutil

src = "checkpoints"  # your source folder
dst = "/home/user/.cache/torch/hub/checkpoints"

# Create destination folder if it doesn't exist
os.makedirs(dst, exist_ok=True)

# Copy each item from src → dst
for item in os.listdir(src):
    s = os.path.join(src, item)
    d = os.path.join(dst, item)

    if os.path.isdir(s):
        # Copy directory
        shutil.copytree(s, d, dirs_exist_ok=True)
    else:
        # Copy file
        shutil.copy2(s, d)

print("✓ Done copying checkpoints!")

import spaces
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
# preload the CNN component

ctypes.CDLL("/usr/local/lib/python3.10/site-packages/nvidia/cudnn/lib/libcudnn_cnn.so.9")

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
snapshot_download("IndexTeam/IndexTTS-2", local_dir=os.path.join(current_dir,"checkpoints"))

dnr_model = tigersound.look2hear.models.TIGERDNR.from_pretrained("JusperLee/TIGER-DnR").to("cuda").eval()


from indextts.infer_v2 import IndexTTS2

MODE = 'local'
tts = IndexTTS2(model_dir="./checkpoints",
                cfg_path=os.path.join("./checkpoints", "config.yaml"),
                use_fp16=True,
                use_deepspeed=False,
                use_cuda_kernel=False,
                )


os.environ["PROCESSED_RESULTS"] = f"{os.getcwd()}/proprocess_results"

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
            # We’ll try a few safe ways to get the final filepath.
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
    """
    text = text.strip()
    chunks = []

    while len(text) > max_chars:
        # Try to split at the last sentence end before max_chars
        split_at = max(
            text.rfind(". ", 0, max_chars),
            text.rfind("! ", 0, max_chars),
            text.rfind("? ", 0, max_chars),
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
MODEL_SIZE = "medium"            # e.g. "small", "medium", "large-v2"
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


def transcribe_segment(whisper_model, samples: np.ndarray) -> str:
    """Transcribe+translate a single segment with faster-whisper."""
    segment_text_parts = []


    segments, info = whisper_model.transcribe(
        samples,
        beam_size=1,
        vad_filter=False,                # diarization already detected speech
        condition_on_previous_text=True,  # independent segments
        task="translate",                # translate to English
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
):
    """
    Transcribe+translate a single diarization segment, returning a
    list of word dicts with absolute timestamps.
    """
    words_out = []

    segments, info = whisper_model.transcribe(
        samples,
        beam_size=1,
        vad_filter=False,                  # diarization already detected speech
        condition_on_previous_text=False,  # better for hard cuts / segments
        task="translate",
        word_timestamps=True,
    )

    for seg in segments:
        if not seg.words:
            continue
        for w in seg.words:
            words_out.append(
                {
                    "start": offset_sec + float(w.start),
                    "end": offset_sec + float(w.end),
                    "text": w.word,
                    "speaker": speaker,
                }
            )

    return words_out

def words_to_subtitles(words, max_seconds: float = 10.0):
    """
    Group word-level timings into SRT subtitles, each up to max_seconds long,
    cutting ONLY at word boundaries, AND never mixing speakers in the same subtitle.
    Whenever the speaker changes, we close the current subtitle and start a new one.

    Expects each word dict to have:
      - "start" (float, seconds)
      - "end"   (float, seconds)
      - "text"  (str)
      - "speaker" (str or None)
    """
    # sort just in case
    words = sorted(words, key=lambda w: w["start"])

    subtitles = []
    current_words = []
    current_start = None
    current_speaker = None

    index = 1

    for w in words:
        w_start = w["start"]
        w_end = w["end"]
        w_speaker = w.get("speaker")

        if current_start is None:
            # start first subtitle
            current_start = w_start
            current_words = [w]
            current_speaker = w_speaker
            continue

        speaker_changed = (w_speaker != current_speaker)
        duration_if_added = w_end - current_start
        exceeds_max = duration_if_added > max_seconds

        # If adding this word would:
        #   - exceed max_seconds, OR
        #   - cross into a different speaker,
        # then we close the current subtitle and start a new one.
        if (speaker_changed or exceeds_max) and current_words:
            text = " ".join(x["text"] for x in current_words).strip()
            sub_start = current_start
            sub_end = current_words[-1]["end"]

            subtitles.append(
                srt.Subtitle(
                    index=index,
                    start=timedelta(seconds=sub_start),
                    end=timedelta(seconds=sub_end),
                    content=text,
                )
            )
            index += 1

            # start new subtitle from this word
            current_start = w_start
            current_words = [w]
            current_speaker = w_speaker
        else:
            current_words.append(w)

    # flush last subtitle
    if current_words:
        text = " ".join(x["text"] for x in current_words).strip()
        sub_start = current_start
        sub_end = current_words[-1]["end"]
        subtitles.append(
            srt.Subtitle(
                index=index,
                start=timedelta(seconds=sub_start),
                end=timedelta(seconds=sub_end),
                content=text,
            )
        )

    return subtitles

def build_srt(segments: List[Dict], audio_wav: str, out_srt_path: str):
    """
    Generate SRT file from diarized segments and audio,
    using word-level timestamps and grouping into ~10s subtitles.
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

        # get words for this diar segment, with absolute times
        seg_words = transcribe_segment_words(
            whisper_model,
            samples,
            offset_sec=start_sec,
            speaker=speaker,
        )

        all_words.extend(seg_words)
        print(f"Diar segment {i} ({speaker}): {len(seg_words)} words")

    # group words into ≤10s subtitles, word aligned
    subtitles = words_to_subtitles(all_words, max_seconds=10.0)

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


def translate_video(video_file, url_or_path, duration, session_id=None, progress=gr.Progress(track_tqdm=True)):

    if video_file == None:
        url_or_path = url_or_path
    else:
        url_or_path = video_file
    video_file = resolve_video_input(url_or_path)
    return process_video(video_file, False, duration, session_id, progress)


def translate_lipsync_video(video_file, url_or_path, duration, session_id=None, progress=gr.Progress(track_tqdm=True)):

    if video_file == None:
        url_or_path = url_or_path
    else:
        url_or_path = video_file
        
    video_file = resolve_video_input(url_or_path)
    return process_video(video_file, True, duration, session_id, progress)


def run_example(video_file, allow_lipsync, duration, session_id = None, progress=gr.Progress(track_tqdm=True)):

    with timer("processed"):
        result = process_video(video_file, allow_lipsync, duration, session_id, progress)

    return result

def get_duration(video_file, allow_lipsync, duration, session_id, progress):

    if allow_lipsync:
        return (60 + 30 * (duration) // 5)
    else:
        return (60 + 20 * (duration) // 30)
        
@spaces.GPU(duration=get_duration)
def process_video(video_file, allow_lipsync, duration, session_id = None, progress=gr.Progress(track_tqdm=True)):
    """
    Gradio callback:
    - video_file: temp file object/path from Gradio
    - returns path to generated SRT file (for download)
    """
    import onnxruntime as ort

    if session_id == None:
        session_id = uuid.uuid4().hex

    output_dir = os.path.join(os.environ["PROCESSED_RESULTS"], session_id)
    os.makedirs(output_dir, exist_ok=True)

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

    # 3. Build SRT from diarized segments + whisper
    with timer("Generating srt"):
        build_srt(segments, audio_16k_wav, srt_path)

    # ---- ORIGINAL SRT (used for TTS) ----
    with open(srt_path, "r", encoding="utf-8") as f:
        srt_data = f.read()

    subtitles = list(srt.parse(srt_data))

    # Keep this list as-is for TTS timing
    tts_subtitles = subtitles

    # ---- CREATE 10s-MAX SRT FOR DOWNLOAD ----
    max10_subtitles = tts_subtitles
    # max10_subtitles = split_subtitles_max_duration(subtitles, max_seconds=10.0)

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
            spk_prompts = []      # paths to src_prompt_*.wav
            texts = []            # subtitle texts for this batch
            out_paths = []        # where IndexTTS2 will save generated wavs
            starts_ms = []        # for overlaying later
            target_ms_list = []   # per-subtitle target durations
            batch_ms_sum = 0

            batch_start = idx

            # ---- fill one batch until we hit ~MAX_BATCH_MS ----
            while idx < num_subs:
                sub = tts_subtitles[idx]

                start_ms = int(sub.start.total_seconds() * 1000)
                end_ms = int(sub.end.total_seconds() * 1000)
                target_ms = max(end_ms - start_ms, 0)

                # If adding this subtitle would exceed the limit and we already
                # have something in the batch, stop and process the current batch.
                if batch_ms_sum + target_ms > MAX_BATCH_MS and len(target_ms_list) > 0:
                    break

                global_idx = idx

                # 1) prompt audio for this subtitle
                src_chunk = original_audio[start_ms:end_ms]
                src_prompt_path = os.path.join(output_dir, f"src_prompt_{global_idx}.wav")
                src_chunk.export(src_prompt_path, format="wav")

                # 2) text + output path
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

            # --- call batched TTS once for this batch ---
            do_sample = True
            top_p = 0.8
            top_k = 30
            temperature = 0.8
            length_penalty = 0.0
            num_beams = 3
            repetition_penalty = 10.0
            max_mel_tokens = 1500

            # You could compute some aggregate target_length_ms here if your API supports it,
            # e.g. avg or max(target_ms_list). For now, keep None as before.
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
                max_text_tokens_per_segment=120,
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

            # --- read generated wavs and overlay them ---
            for local_idx, out_path in enumerate(tts_outputs):
                start_ms = starts_ms[local_idx]

                seg = AudioSegment.from_file(out_path, format="wav")
                seg = seg - 2
                timeline = timeline.overlay(seg, position=start_ms)

                # cleanup
                os.remove(out_path)
                os.remove(spk_prompts[local_idx])

    # -------------------------------------------------------
    # Bring back original dialog in the *gaps* (grunts, etc.)
    # -------------------------------------------------------
    # Load separated dialog track
    dialog = AudioSegment.from_file(vocal_wav)

    # Make sure it matches the TTS timeline parameters
    dialog = dialog.set_frame_rate(timeline.frame_rate).set_channels(timeline.channels)

    total_len_ms = len(timeline)

    # Collect speech regions from subtitles (approximate "where TTS will speak")
    speech_regions = []
    for sub in tts_subtitles:
        start_ms = int(sub.start.total_seconds() * 1000)
        end_ms = int(sub.end.total_seconds() * 1000)
        # clamp to track length
        start_ms = max(0, min(start_ms, total_len_ms))
        end_ms = max(0, min(end_ms, total_len_ms))
        if end_ms > start_ms:
            speech_regions.append((start_ms, end_ms))

    # Merge overlapping/adjacent regions
    speech_regions.sort()
    merged = []
    for s, e in speech_regions:
        if not merged:
            merged.append([s, e])
        else:
            last_s, last_e = merged[-1]
            if s <= last_e:  # overlap or touch
                merged[-1][1] = max(last_e, e)
            else:
                merged.append([s, e])

    # Compute the complement: regions where there's NO subtitle (gaps)
    gaps = []
    cursor = 0
    for s, e in merged:
        if cursor < s:
            gaps.append((cursor, s))
        cursor = max(cursor, e)
    if cursor < total_len_ms:
        gaps.append((cursor, total_len_ms))

    # Overlay original dialog only in those gaps
    MIN_GAP_MS = 10  # ignore ultra-tiny gaps

    for g_start, g_end in gaps:
        if g_end - g_start < MIN_GAP_MS:
            continue

        # Extract that piece of the original dialog
        original_chunk = dialog[g_start:g_end]
        original_chunk = original_chunk + 6

        timeline = timeline.overlay(original_chunk, position=g_start)


    video_in = video_path
    audio_in = output_dir + "/final_output.wav"
    audio_16k_in = output_dir + "/final_16k_output.wav"
    
    # ---------- 5. Mix background + new TTS vocal ----------
    
    if background_wav is not None:
        eff = AudioSegment.from_file(effect_wav)
        bg = AudioSegment.from_file(background_wav)

        
    
        # If background is shorter than the TTS timeline, loop it
        if len(eff) < len(timeline):
            loops = math.ceil(len(timeline) / len(eff))
            eff = eff * loops
                    
        if len(bg) < len(timeline):
            loops = math.ceil(len(timeline) / len(bg))
            bg = bg * loops


    
        # Cut or match to TTS length
        eff = eff[:len(timeline)]
        bg = bg[:len(timeline)]
        
    
        bg = bg + 2
        eff = eff + 2
    
        eff_timeline = eff.overlay(timeline)
        final_audio = bg.overlay(eff_timeline)
        final_16k_audio = timeline.set_frame_rate(16000).set_channels(1)
    else:
        # Fallback: no background found, just use TTS
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
        "-y",               # overwrite output file
        "-i", lipsynced_video,     # input video
        "-i", audio_in_upsampled,     # new audio
        "-c:v", "copy",     # do not re-encode video
        "-map", "0:v:0",    # take video from input 0
        "-map", "1:a:0",    # take audio from input 1
        "-shortest",        # stop when either track ends
        video_out,
    ]


    subprocess.run(cmd, check=True)


    # IMPORTANT: return the 10s-max SRT for download
    return video_out, srt_10s_path, audio_16k_in



css = """
    #col-container {
        margin: 0 auto;
        max-width: 1600px;
    }
    #modal-container {
    width: 100vw;            /* Take full viewport width */
    height: 100vh;           /* Take full viewport height (optional) */
    display: flex;           
    justify-content: center; /* Center content horizontally */
    align-items: center;     /* Center content vertically if desired */
    }
    #modal-content {
    width: 100%;
    max-width: 700px;         /* Limit content width */
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
    background-color: #ffd6ff;  /* light pink background */
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
    color: #4b2ab5; /* dark purple text for unselected */
    padding: 6px 22px;
    border-radius: 9999px;
    cursor: pointer;
    transition: color 0.25s ease;
    }
    /* Moving highlight */
    .toggle-highlight {
    position: absolute;
    top: 4px;
    left: 4px;
    width: calc(50% - 4px);
    height: calc(100% - 8px);
    background-color: #4b2ab5; /* dark purple background */
    border-radius: 9999px;
    transition: transform 0.25s ease;
    z-index: 1;
    }
    /* When "True" is checked */
    #true:checked ~ label[for="true"] {
    color: #ffd6ff; /* light pink text */
    }
    /* When "False" is checked */
    #false:checked ~ label[for="false"] {
    color: #ffd6ff; /* light pink text */
    }
    /* Move highlight to right side when False is checked */
    #false:checked ~ .toggle-highlight {
    transform: translateX(100%);
    }
    """

def _fmt_seconds(sec: int) -> str:
    sec = int(sec)
    return f"{sec}s"

def compute_etas(duration_value: int):
    # get_duration signature: (video_file, allow_lipsync, duration, session_id, progress)
    # We only need allow_lipsync + duration; pass placeholders for the rest.
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

with gr.Blocks(css=css) as demo:

    session_state = gr.State()
    demo.load(start_session, outputs=[session_state])

    with gr.Column(elem_id="col-container"):
        gr.HTML(
            """
            <div style="text-align: center;">
                <p style="font-size:16px; display: inline; margin: 0;">
                    Translate and lipsync your clips from any language to English
                </p>
            </div>
            <div style="text-align: center;">
                <p style="font-size:16px; display: inline; margin: 0;">
                    <strong>OutofLipSync</strong>
                </p>
                <p style="font-size:16px; display: inline; margin: 0;">
                    -- HF Space By:
                </p>
                <a href="https://huggingface.co/alexnasa" style="display: inline-block; vertical-align: middle; margin-left: 0.5em;">
                    <img src="https://img.shields.io/badge/🤗-Follow Me-yellow.svg">
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

                duration = gr.Slider(5, 120, 10, step=1, label="Duration(s)")

                uncached_examples = gr.Examples(                    
                    examples=[ 

                        [
                            "assets/popup-2.mp4",
                        ],
                        
                        [
                            "assets/sofia-esp.mp4",
                        ],

                        [
                            "assets/alba-port.mp4",
                        ],

                        [
                            "assets/lena-de.mp4",
                        ],
                    ],
                    inputs=video_input,
                    )

            with gr.Column(elem_id="step-column"):
                gr.HTML("""
                <div>
                    <span style="font-size: 24px;">2. Translate + 💋 </span><br>
                </div>
                """)

                video_output = gr.Video(label="Output", height=512)
                lipsync = gr.Checkbox(label="Lipsync", value=False, visible=False)

                eta_translate_md = gr.Markdown("")
                eta_lipsync_md = gr.Markdown("")

                translate_btn = gr.Button("🤹‍♂️ Translate")
                translate_lipsync_btn = gr.Button("🤹‍♂️ Translate + 💋 Lipsync", variant='primary', elem_classes="button-gradient")
        
            with gr.Column(elem_id="step-column"):
                gr.HTML("""
                <div>
                    <span style="font-size: 24px;">Lipsynced Examples </span><br>
                </div>
                """)
                vocal_16k_output = gr.File(label="Vocal 16k", visible=False)
                srt_output = gr.File(label="Download translated diarized SRT", visible=False)

                cached_examples = gr.Examples(                    
                    examples=[ 
            
                        [
                            "assets/monica-ita.mp4",
                            True,
                            5
                        ],

                        [
                            "assets/koji.mp4",
                            False,
                            60
                        ],

                        [
                            "assets/elena-es.mp4",
                            True,
                            10
                        ],

                        [
                            "assets/ana-es.mp4",
                            True,
                            10
                        ],
                        
                        [
                            "assets/spanish-2.mp4",
                            True,
                            5
                        ],

                        [
                            "assets/italian.mp4",
                            True,
                            5
                        ],

                        [
                            "assets/alica-por-2.mp4",
                            True,
                            10
                        ],

            
                    ],
                    fn=run_example,
                    inputs=[video_input, lipsync, duration],
                    outputs=[video_output, srt_output, vocal_16k_output],
                    cache_examples=True
                    )
        

    translate_btn.click(
        fn=translate_video,
        inputs=[video_input, url_in, duration, session_state],
        outputs=[video_output, srt_output, vocal_16k_output],
    )
    
    translate_lipsync_btn.click(
        fn=translate_lipsync_video,
        inputs=[video_input, url_in, duration, session_state],
        outputs=[video_output, srt_output, vocal_16k_output],
    )

    duration.change(
        fn=compute_etas,
        inputs=[duration],
        outputs=[eta_translate_md, eta_lipsync_md],
    )



if __name__ == "__main__":
    demo.unload(cleanup)
    demo.queue()
    demo.launch(ssr_mode=False)