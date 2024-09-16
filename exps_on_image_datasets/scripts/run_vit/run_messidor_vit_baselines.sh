for lr in 0.01 0.001 0.0001
do
python train_label_noise.py --config configs/config_constraint_messidor_vit.json \
    --model VisionTransformer --is_vit --img_size 224 --vit_type ViT-B_16 --vit_pretrained_dir pretrained/imagenet21k_ViT-B_16.npz \
    --lr $lr --runs 1 --device 2
done