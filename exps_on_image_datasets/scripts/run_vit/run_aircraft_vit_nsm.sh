for sigma in 0.01 0.008 0.005 0.012
do
python train_label_noise.py --config configs/config_constraint_aircrafts.json \
    --model VisionTransformer --is_vit --img_size 224 --vit_type ViT-B_16 --vit_pretrained_dir pretrained/imagenet21k_ViT-B_16.npz \
    --lr 0.0001 --runs 1 --device 3 \
    --train_nsm --use_neg --nsm_lam 0 --num_perturbs 1 --nsm_sigma $sigma
done

python train_label_noise.py --config configs/config_constraint_aircrafts.json \
    --model ResNet101 \
    --lr 0.0001 --runs 1 --device 3

#--is_vit --img_size 224 --vit_type ViT-B_16 --vit_pretrained_dir pretrained/imagenet21k_ViT-B_16.npz \