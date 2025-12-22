# ByteDance/LatentSync
from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
import torch

pipeline = LipsyncPipeline(
    scheduler=None,
    vae=None,
    audio_encoder=None,
    unet=None,
).to("cuda")

def apply_lipsync(video_input_path, audio_path, video_out_path):

    torch.manual_seed(1234)

    print(f"Initial seed: {torch.initial_seed()}")

    pipeline(
        video_path=video_input_path,
        audio_path=audio_path,
        video_out_path=video_out_path,
        video_mask_path=video_out_path.replace(".mp4", "_mask.mp4"),
        num_frames=16,
        num_inference_steps=20,
        guidance_scale=1.0,
        weight_dtype=torch.float16,
        width=256,
        height=256,
    )
    torch.cuda.empty_cache()

    return video_out_path
