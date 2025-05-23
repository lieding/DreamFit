

export CUDA_VISIBLE_DEVICES=0
python3 inference_dreamfit_flux_i2i.py \
     --config configs/inference/inference_dreamfit_flux_i2i.yaml \
    "$@"