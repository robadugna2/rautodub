# ✅ ETHIOPIAN DUBBING — Quick Setup Checklist

## You Need to Prepare (BEFORE running)

### 1. Upload Your Fine-tuned IndexTTS2 Models to HuggingFace
```bash
# Upload each model to a HF repo:
# Option A: Single multi-language model
huggingface-cli upload robadugna/rtts2 ./your-amharic-oromo-tigrinya-model/

# Option B: Separate models (recommended)
huggingface-cli upload YOUR_USERNAME/IndexTTS-2-amharic  ./your-amharic-model/
huggingface-cli upload YOUR_USERNAME/IndexTTS-2-tigrinya ./your-tigrinya-model/
huggingface-cli upload YOUR_USERNAME/IndexTTS-2-oromo    ./your-oromo-model/
```

### 2. Update `modified_app.py` — Find & Replace These Lines:
```python
# Line ~87-89: Replace with YOUR actual HF repo names
TTS_MODEL_REPOS = {
    "am":  "YOUR_HF_USERNAME/IndexTTS-2-amharic",   # ← CHANGE THIS
    "tir": "YOUR_HF_USERNAME/IndexTTS-2-tigrinya",   # ← CHANGE THIS
    "om":  "YOUR_HF_USERNAME/IndexTTS-2-oromo",      # ← CHANGE THIS
}

# Line ~109: If you have a SINGLE model for all languages, replace with:
# TTS_MODEL_REPOS = {
#     "am":  "robadugna/rtts2",
#     "tir": "robadugna/rtts2",
#     "om":  "robadugna/rtts2",
# }
```

### 3. Translation API — No Download Needed!
Translation is done via the free Mapiz API (`https://dev-mapiz.pantheonsite.io/ymigxf/Api/`).
No local model download required — saves ~7GB VRAM vs NLLB!

**Test it first:**
```bash
python modified_app.py --test-api
```

### 4. GPU Requirements (UPDATED — lower now!)
| Component | VRAM Estimate |
|-----------|---------------|
| IndexTTS2 (per model) | ~4 GB |
| Whisper large-v3 | ~3 GB |
| TIGER-DnR | ~2 GB |
| LatentSync (if lip-sync) | ~4 GB |
| Translation API | 0 GB (remote) |
| **Total (no lipsync)** | **~9 GB** |
| **Total (with lipsync)** | **~13 GB** |

---

## What Changed in the Code (Summary)

| # | What | Original | Modified |
|---|------|----------|----------|
| 1 | Translation | Whisper `task="translate"` (→ English) | Whisper ASR + Mapiz Chat Completion API (→ AM/TIR/OM) |
| 2 | TTS model | `IndexTeam/IndexTTS-2` (English) | Your fine-tuned single multilingual model |
| 3 | Whisper size | `medium` | `large-v3` (better Ethiopian recognition) |
| 4 | UI | No language selector | Dropdown: አማርኛ / ትግርኛ / Afaan Oromoo |
| 5 | Subtitles | English SRT | Amharic/Tigrinya/Oromo SRT |

---

## How the Modified Pipeline Works

```
Input Video (any language)
    │
    ├─► FFmpeg extract audio
    │
    ├─► TIGER-DnR: separate → vocals / effects / background
    │
    ├─► pyannote: speaker diarization (who speaks when)
    │
    ├─► Whisper large-v3: ASR → transcribe source text
    │         (task="transcribe", NOT "translate")
    │
    ├─► Mapiz API: translate subtitle text → Amharic/Tigrinya/Oromo
    │         (done at subtitle level to avoid repetitions)
    │
    ├─► IndexTTS2 (your fine-tuned model): 
    │     speak the translated text in cloned voice
    │
    ├─► Mix: TTS voice + original effects + original background
    │
    ├─► FlashSR: upsample audio to 48kHz
    │
    ├─► [Optional] LatentSync: lip-sync video to new audio
    │
    └─► Output: dubbed video + SRT subtitle file
```

---

## Running It
```bash
cd OutofLipSync
# Rename modified to active
mv app.py app_original.py
mv modified_app.py app.py

# Run locally
python app.py

# Or deploy to HuggingFace Spaces
# Just push to your HF Space repo
```

## Testing Tips
1. Start with a short clip (5-10 seconds) to test
2. Try Amharic first (your best model with 1.1k hrs)
3. Check if NLLB translation quality is acceptable — if not, you can swap in:
   - Google Translate API
   - Your own translation model
   - A fine-tuned NLLB for your specific language pair
