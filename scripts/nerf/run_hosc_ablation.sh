#!/bin/bash

# examples=("chair")
examples=("drums" "chair" "ficus" "hotdog" "lego" "materials" "mic" "ship")
betas=(0.1 0.3 0.5 0.8 1.0 3.0)

for beta in "${betas[@]}"
do
    for example in "${examples[@]}"
    do
        OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 python ./main_nerf.py ../data/nerf/nerf_synthetic/${example} \
        --nn hosc --lr 2e-4 --iter 37500 --downscale 4 \
        --trainskip 4 \
        --beta $beta \
        --num_layers 4 --hidden_dim 182 --geo_feat_dim 182 --num_layers_color 4 --hidden_dim_color 182 \
        --workspace logs/hosc_ablation/${example}_hosc_${beta} \
        -O --bound 1 --scale 0.8 --dt_gamma 0
    done
done
wait