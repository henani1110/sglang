#!/usr/bin/env bash
set -euo pipefail
cd /home/chx/projects/sglang-h3-exact-lora-20261029
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=1
export CUDA_HOME=/usr/local/cuda-13.0
export PATH="/usr/local/cuda-13.0/bin:/home/chx/projects/sglang/.venv/bin:$PATH"
export PYTHONPATH="/home/chx/experiments/minimax-h3-sm89-kernel-20261030/sage2/site:/home/chx/projects/sglang-h3-exact-lora-20261029/python:/home/chx/experiments/minimax-h3-comfyweights-8step-20261029/compat"
export SGLANG_DIFFUSION_SYNC_STAGE_PROFILING=1
exec /home/chx/projects/sglang/.venv/bin/sglang serve \
  --model-type diffusion \
  --model-path MiniMaxAI/MiniMax-H3 \
  --model-variant fl2va \
  --num-gpus 1 \
  --component-weights-paths.transformer /home/flyvideo/ComfyUI/models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors \
  --component-weights-paths.text_encoder /home/flyvideo/ComfyUI/models/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors \
  --component-weights-paths.audio_vae /home/flyvideo/ComfyUI/models/vae/minimax_h3_audio_vae_fp32.safetensors \
  --lora-path /home/flyvideo/ComfyUI/models/loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors \
  --lora-scale 1.0 \
  --lora-merge-mode auto \
  --attention-backend fa \
  --component-attention-backends.transformer sage_attn \
  --performance-mode memory \
  --layerwise-offload-components dit,text_encoder \
  --dit-offload-prefetch-size 1 \
  --dit-layerwise-resident-layers 0 \
  --enable-torch-compile false \
  --warmup-mode off \
  --warmup-steps 1 \
  --host 127.0.0.1 \
  --port 30014
