export CUDA_VISIBLE_DEVICES=0

python3 inference_dreamfit_sd15_i2i.py \
    --config configs/inference/inference_dreamfit_sd15_i2i.yaml \
    --vae_ckpt pretrained_models/sd-vae-ft-mse/diffusion_pytorch_model.bin \
    --ref_model pretrained_models/sd15_i2i.ckpt \
    "$@"