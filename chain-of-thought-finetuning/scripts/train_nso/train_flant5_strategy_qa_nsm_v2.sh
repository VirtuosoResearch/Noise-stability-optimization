# python custom_train.py --dataset_key commonsense_qa --model_key gpt2 --train_key ft_cot \
#     --preset_key ft_cot --devices 2 --batch_size 8 --inference_batch_size 32 \
#     --runs 2


for alpha in 0.01 0.012 0.014
do
python custom_train.py --dataset_key commonsense_qa --model_key gpt2 --train_key ft_cot \
    --preset_key ft_cot --devices 2 --batch_size 8 --inference_batch_size 32 \
    --runs 2\
    --train_nsm --nsm_use_neg --nsm_lam 0 --nsm_num_perturbs 1 --nsm_sigma $alpha --epochs 5
done