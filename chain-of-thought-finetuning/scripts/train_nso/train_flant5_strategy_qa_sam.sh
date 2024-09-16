# python custom_train.py --dataset_key strategy_qa --model_key gpt2 --train_key ft_cot \
#     --preset_key ft_cot_t70_64aug --devices 2 --batch_size 8 --inference_batch_size 32 \
#     --runs 2

for rho in 0.01 0.02 0.05
do
python custom_train.py --dataset_key commonsense_qa --model_key gpt2 --train_key ft_cot \
    --preset_key ft_cot --devices 2 --batch_size 8 --inference_batch_size 32 \
    --runs 3\
    --train_sam --sam_rho $rho --epochs 5
done

for rho in 0.01 0.02 0.05
do
python custom_train.py --dataset_key strategy_qa --model_key gpt2 --train_key ft_cot \
    --preset_key ft_cot_t70_64aug --devices 2 --batch_size 8 --inference_batch_size 32 \
    --runs 2\
    --train_sam --sam_rho $rho --epochs 5
done
