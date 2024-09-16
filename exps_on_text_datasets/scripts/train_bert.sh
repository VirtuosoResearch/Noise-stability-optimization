# python train_glue_label_noise.py --config configs/config_glue.json --task_name mrpc \
#     --epochs 10 --runs 1 --device 3 --noise_rate 0 --model_name_or_path bert-base-uncased --lr 1e-5

# python train_glue_label_noise.py --config configs/config_glue.json --task_name mrpc \
#     --epochs 1 --runs 1 --device 3 --noise_rate 0 --model_name_or_path bert-base-uncased\
#     --train_nsm --train_nsm --use_neg --nsm_lam 0 --num_perturbs 1 --nsm_sigma 0.01 --lr 1e-5

python train_glue_label_noise.py --config configs/config_glue.json --task_name mrpc \
    --epochs 10 --runs 1 --device 2 --noise_rate 0 --model_name_or_path bert-base-uncased\
    --train_nsm --train_nsm --nsm_lam 0 --num_perturbs 2 --nsm_sigma 0.01 --lr 1e-5