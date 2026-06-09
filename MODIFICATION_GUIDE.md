# OutofLipSync → Ethiopian Languages (AM/TIR/OM) Modification Guide

## Overview
Transform OutofLipSync from "any language → English" to "any language → Amharic / Tigrinya / Oromo"

## The 5 Key Changes

### 1. 🧠 Translation Pipeline (THE BIGGEST CHANGE)
**Current:** Whisper `task="translate"` → translates everything to English automatically
**Problem:** Whisper can ONLY translate TO English. It cannot translate to Amharic/Tigrinya/Oromo.

**Solution — Two-step approach:**
- Step A: Whisper ASR (transcribe in source language) → get text
- Step B: Free Mapiz API → translate to Amharic/Tigrinya/Oromo

**Translation: Free API (no model download needed!)**
| Component | Details |
|-----------|---------|
| API URL | `https://dev-mapiz.pantheonsite.io/ymigxf/Api/` |
| Format | OpenAI-compatible chat completions |
| Cost | **FREE** |
| VRAM | **0 GB** (runs on remote server) |
| Languages | Any → Amharic, Tigrinya, Oromo |

### 2. 🎙️ TTS Model Swap
**Current:** Downloads `IndexTeam/IndexTTS-2` (base model, English-centric)
**Your change:** Point to your fine-tuned models

```python
# Option A: If you have ONE model fine-tuned on all 3 languages
snapshot_download("robadugna/rtts2", local_dir="./checkpoints")


### 3. 🗣️ Whisper Model Size
**Current:** `MODEL_SIZE = "medium"`
**Change to:** `MODEL_SIZE = "large-v3"` 
**Why:** Large-v3 has much better coverage of Amharic, Tigrinya, and other Ethiopian languages.
The "medium" model has poor recognition for these languages.

### 4. 🖥️ UI: Add Language Selector
Add a dropdown for target language: `["Amharic (አማርኛ)", "Tigrinya (ትግርኛ)", "Oromo (Afaan Oromoo)"]`

### 5. 📝 Subtitle Text Handling
Ethiopic script (Ge'ez) uses different word boundaries than Latin. The `split_text_into_chunks` 
and subtitle grouping functions should work fine since they split on spaces, and all three 
languages use spaces between words.

---

## Files to Modify

| File | Changes |
|------|---------|
| `app.py` | Translation pipeline, TTS loading, UI language selector, Whisper model size |
| `requirements.txt` | Add `transformers`, `sentencepiece` (already there), add `protobuf` for NLLB |
| `README.md` | Update description |
| `lipsync.py` | NO changes needed |

---

## Quick Start
```bash
# 1. Clone this repo
git clone https://huggingface.co/spaces/alexnasa/OutofLipSync

# 2. Apply the modifications (see modified_app.py)

# 3. Upload your fine-tuned IndexTTS2 checkpoints

# 4. Run
python app.py
```
