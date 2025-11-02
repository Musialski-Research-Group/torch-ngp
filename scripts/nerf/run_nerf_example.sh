OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 python ./main_nerf.py ../data/nerf/nerf_synthetic/drums \
    --nn finer --lr 2e-4 --iter 37500 --downscale 4 \
    --trainskip 4 \
    --num_layers 4 --hidden_dim 182 --geo_feat_dim 182 --num_layers_color 4 --hidden_dim_color 182 \
    --workspace logs/drums_finer \
    -O --bound 1 --scale 0.8 --dt_gamma 0