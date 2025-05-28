export FLUX_DEV="/home/featurize/work/app/comfyui/ComfyUI/models/unet/flux1-dev-fp8.safetensors"
export AE="/home/featurize/work/app/comfyui/ComfyUI/models/vae/ae.safetensors"

export CUDA_VISIBLE_DEVICES=0
python3 inference_dreamfit_flux_tryon.py \
    --config configs/inference/inference_dreamfit_flux_tryon.yaml \
    "$@"