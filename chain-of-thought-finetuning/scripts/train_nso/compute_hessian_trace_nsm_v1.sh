python compute_hessian_traces.py --dataset_key strategy_qa --model_key gpt2 --train_key ft_cot --preset_key ft_cot_t70_64aug\
    --load_model_dir gpt2_strategy_qa_ft_cot_t70_64aug_run_0/epoch_epoch=0 --devices 1

python compute_hessian_traces.py --dataset_key strategy_qa --model_key gpt2 --train_key ft_cot --preset_key ft_cot_t70_64aug\
    --load_model_dir gpt2_strategy_qa_ft_cot_t70_64aug_sam_rho_0.01/epoch_epoch=4 --devices 1

python compute_hessian_traces.py --dataset_key strategy_qa --model_key gpt2 --train_key ft_cot --preset_key ft_cot_t70_64aug\
    --load_model_dir gpt2_strategy_qa_ft_cot_t70_64aug_sam_rho_0.02/epoch_epoch=4 --devices 1

python compute_hessian_traces.py --dataset_key strategy_qa --model_key gpt2 --train_key ft_cot --preset_key ft_cot_t70_64aug\
    --load_model_dir gpt2_strategy_qa_ft_cot_t70_64aug_sam_rho_0.05/epoch_epoch=4 --devices 1

python compute_hessian_traces.py --dataset_key strategy_qa --model_key gpt2 --train_key ft_cot --preset_key ft_cot_t70_64aug\
    --load_model_dir gpt2_strategy_qa_ft_cot_t70_64aug_nsm_sigma_0.01/epoch_epoch=4 --devices 1 

python compute_hessian_traces.py --dataset_key strategy_qa --model_key gpt2 --train_key ft_cot --preset_key ft_cot_t70_64aug\
    --load_model_dir gpt2_strategy_qa_ft_cot_t70_64aug_nsm_sigma_0.008/epoch_epoch=4 --devices 1

python compute_hessian_traces.py --dataset_key strategy_qa --model_key gpt2 --train_key ft_cot --preset_key ft_cot_t70_64aug\
    --load_model_dir gpt2_strategy_qa_ft_cot_t70_64aug_nsm_sigma_0.012/epoch_epoch=4 --devices 1