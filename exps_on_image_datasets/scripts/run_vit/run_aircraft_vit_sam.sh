python train_label_noise.py --config configs/config_constraint_aircrafts.json \
    --model VisionTransformer --is_vit --img_size 224 --vit_type ViT-B_16 --vit_pretrained_dir pretrained/imagenet21k_ViT-B_16.npz \
    --lr 0.0001 --runs 3 --device 2

for rho in 0.01 0.02 0.05
do
python train_label_noise.py --config configs/config_constraint_aircrafts.json \
    --model VisionTransformer --is_vit --img_size 224 --vit_type ViT-B_16 --vit_pretrained_dir pretrained/imagenet21k_ViT-B_16.npz \
    --lr 0.0001 --runs 3 --device 2 --train_sam --sam_rho $rho
done