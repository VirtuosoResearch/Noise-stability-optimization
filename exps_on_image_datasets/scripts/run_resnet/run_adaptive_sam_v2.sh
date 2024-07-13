python train_label_noise.py --config configs/config_constraint_aircrafts.json \
    --model ResNet34 --lr 0.0001 --runs 1 --device 3 --train_sam --sam_rho 0.1  --sam_adaptive

python train_label_noise.py --config configs/config_constraint_birds.json \
    --model ResNet34 --lr 0.0001 --runs 1 --device 3 --train_sam --sam_rho 0.1  --sam_adaptive

python train_label_noise.py --config configs/config_constraint_messidor.json \
    --model ResNet34 --lr 0.0001 --runs 1 --device 3 --train_sam --sam_rho 0.1  --sam_adaptive

python train_label_noise.py --config configs/config_constraint_aptos.json \
    --model ResNet34 --lr 0.0001 --runs 1 --device 3 --train_sam --sam_rho 0.1  --sam_adaptive
