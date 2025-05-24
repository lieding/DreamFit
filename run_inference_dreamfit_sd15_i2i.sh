export CUDA_VISIBLE_DEVICES=0

python3 inference_dreamfit_sd15_i2i.py \
    --config configs/inference/inference_dreamfit_sd15_i2i.yaml \
    --vae_ckpt /home/featurize/work/pretrained_models/diffusion_pytorch_model.bin \
    --ref_model /home/featurize/work/pretrained_models/sd15_i2i.ckpt \
    --base_model /home/featurize/work/app/comfyui/ComfyUI/models/checkpoints/epicphotogasm_ultimateFidelity.safetensors \
    "$@"