for lr in 0.02 0.05 0.1 
do
python train_label_noise.py --config configs/config_constraint_aircrafts_sgd.json \
    --model ResNet34 --runs 1 --device 0 --lr $lr
done
