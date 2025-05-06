export FLUX_DEV="pretrained_models/FLUX.1-dev/flux1-dev.safetensors"
export AE="pretrained_models/FLUX.1-dev/ae.safetensors"

export CUDA_VISIBLE_DEVICES=0
python3 inference_dreamfit_flux_tryon.py \
    --config configs/inference/inference_dreamfit_flux_tryon.yaml \
    "$@"