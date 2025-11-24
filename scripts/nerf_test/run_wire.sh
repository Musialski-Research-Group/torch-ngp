#!/bin/bash

# examples=("drums" "chair" "ficus" "hotdog" "lego" "materials" "mic" "ship")
# examples=("lego" "materials" "mic" "ship")
examples=("ship")

for example in "${examples[@]}"
do
    OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 python ./main_nerf.py ../data/nerf/nerf_synthetic/${example} \
    --nn wire --lr 2e-4 --iter 37500 --downscale 1 \
    --trainskip 4 \
    --num_layers 4 --hidden_dim 182 --geo_feat_dim 182 --num_layers_color 4 --hidden_dim_color 182 \
    --test \
    --workspace logs_wire_final/${example}_wire_final \
    -O --bound 1 --scale 0.8 --dt_gamma 0
done
wait