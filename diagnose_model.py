"""
Quick diagnostic for your fine-tuned IndexTTS2 checkpoint.
Run this on your deployment machine (Lightning AI / HF Spaces) where
the checkpoints are downloaded.

Usage:
    python diagnose_model.py
    python diagnose_model.py --checkpoint-dir ./checkpoints
    python diagnose_model.py --checkpoint-dir ./checkpoints_multilingual
"""
import os
import sys
import argparse

def diagnose():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", default=None,
                        help="Path to checkpoint directory")
    args = parser.parse_args()

    # Auto-detect checkpoint dir
    candidates = [args.checkpoint_dir, "./checkpoints_multilingual", "./checkpoints"]
    ckpt_dir = None
    for c in candidates:
        if c and os.path.isdir(c):
            ckpt_dir = c
            break

    if not ckpt_dir:
        print("ERROR: No checkpoint directory found. Use --checkpoint-dir")
        return

    print("=" * 60)
    print(f"  CHECKPOINT DIR: {os.path.abspath(ckpt_dir)}")
    print("=" * 60)

    # --- 1. List files ---
    print("\n[1] FILES:")
    for f in sorted(os.listdir(ckpt_dir)):
        full = os.path.join(ckpt_dir, f)
        if os.path.isfile(full):
            sz = os.path.getsize(full)
            if sz > 1e9:
                print(f"  {f:45s} {sz/1e9:.2f} GB")
            elif sz > 1e6:
                print(f"  {f:45s} {sz/1e6:.1f} MB")
            else:
                print(f"  {f:45s} {sz/1e3:.1f} KB")
        else:
            print(f"  {f + '/':45s} (dir)")

    # --- 2. Inspect GPT checkpoint ---
    print("\n[2] GPT CHECKPOINT INSPECTION:")
    import torch

    gpt_path = os.path.join(ckpt_dir, "gpt.pth")
    latest_path = os.path.join(ckpt_dir, "latest.pth")
    ckpt_file = gpt_path if os.path.exists(gpt_path) else (latest_path if os.path.exists(latest_path) else None)

    if not ckpt_file:
        print("  ERROR: Neither gpt.pth nor latest.pth found!")
        return

    sz_gb = os.path.getsize(ckpt_file) / 1e9
    print(f"  File: {os.path.basename(ckpt_file)} ({sz_gb:.2f} GB)")
    if sz_gb > 5:
        print(f"  WARNING: File is {sz_gb:.1f}GB (training checkpoint?). Official gpt.pth is ~3.5GB.")

    print(f"  Loading...")
    ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=False)

    if isinstance(ckpt, dict):
        print(f"  Top-level keys: {list(ckpt.keys())}")

        # Check for training checkpoint keys
        has_model = 'model' in ckpt
        has_optimizer = 'optimizer' in ckpt or 'optimizer_state_dict' in ckpt
        print(f"  Has 'model' key: {has_model}")
        print(f"  Has optimizer: {has_optimizer}")

        if 'epoch' in ckpt:
            print(f"  Epoch: {ckpt['epoch']}")
        if 'step' in ckpt or 'global_step' in ckpt:
            print(f"  Step: {ckpt.get('step', ckpt.get('global_step'))}")

        # Extract model state dict (same logic as checkpoint.py)
        model_dict = ckpt['model'] if 'model' in ckpt else ckpt

        if isinstance(model_dict, dict):
            # Find critical layers
            print(f"\n  CRITICAL LAYER SHAPES:")
            for k, v in model_dict.items():
                if hasattr(v, 'shape') and ('text_embedding' in k or 'text_head' in k or 'mel_embedding' in k or 'mel_head' in k):
                    print(f"    {k}: {list(v.shape)}")

            # Determine actual number_text_tokens from weights
            for k, v in model_dict.items():
                if 'text_embedding.weight' == k and hasattr(v, 'shape'):
                    actual_vocab = v.shape[0] - 1  # -1 because code does number_text_tokens * types + 1
                    print(f"\n  >>> ACTUAL number_text_tokens from weights: {actual_vocab}")
                    print(f"  >>> config_amharic.yaml says: 28000")
                    if actual_vocab != 28000:
                        print(f"  >>> MISMATCH! Config must be changed to {actual_vocab}")
                    else:
                        print(f"  >>> MATCH! Config is correct.")
        else:
            print(f"  WARNING: 'model' key is type {type(model_dict).__name__}, not dict")
    else:
        print(f"  Raw state_dict (type: {type(ckpt).__name__})")

    # --- 3. Check BPE path resolution ---
    print("\n[3] BPE PATH RESOLUTION TEST:")
    try:
        import yaml
        cfg_path = os.path.join(ckpt_dir, "config_amharic.yaml")
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f)
            bpe_value = cfg.get("dataset", {}).get("bpe_model", "???")
            resolved = os.path.join(ckpt_dir, bpe_value)
            print(f"  config bpe_model value: '{bpe_value}'")
            print(f"  IndexTTS2 resolves to:  '{resolved}'")
            print(f"  File exists at resolved path: {os.path.exists(resolved)}")
            if not os.path.exists(resolved):
                # Check if it exists at the raw value (absolute path case)
                if os.path.exists(bpe_value):
                    print(f"  BUG: File exists at '{bpe_value}' but IndexTTS2 will look at '{resolved}'")
                    print(f"  FIX: Set bpe_model to just the filename: 'am_om_ti_extended.model'")
        else:
            print(f"  config_amharic.yaml not found")
    except Exception as e:
        print(f"  Error: {e}")

    # --- 4. Summary ---
    print("\n" + "=" * 60)
    print("  SHARE THIS OUTPUT so we can pinpoint the exact fixes needed.")
    print("=" * 60)

if __name__ == "__main__":
    diagnose()
