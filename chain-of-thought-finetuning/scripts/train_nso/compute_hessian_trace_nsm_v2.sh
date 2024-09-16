python compute_hessian_traces.py --dataset_key commonsense_qa --model_key gpt2 --train_key ft_cot --preset_key ft_cot\
    --load_model_dir gpt2_commonsense_qa_ft_cot_run_0/epoch_epoch=1 --devices 0 --batch_size 2

python compute_hessian_traces.py --dataset_key commonsense_qa --model_key gpt2 --train_key ft_cot --preset_key ft_cot\
    --load_model_dir gpt2_commonsense_qa_ft_cot_sam_rho_0.01/epoch_epoch=4 --devices 0 --batch_size 2

python compute_hessian_traces.py --dataset_key commonsense_qa --model_key gpt2 --train_key ft_cot --preset_key ft_cot\
    --load_model_dir gpt2_commonsense_qa_ft_cot_sam_rho_0.02/epoch_epoch=4 --devices 0 --batch_size 2

python compute_hessian_traces.py --dataset_key commonsense_qa --model_key gpt2 --train_key ft_cot --preset_key ft_cot\
    --load_model_dir gpt2_commonsense_qa_ft_cot_sam_rho_0.05/epoch_epoch=4 --devices 0 --batch_size 2

python compute_hessian_traces.py --dataset_key commonsense_qa --model_key gpt2 --train_key ft_cot --preset_key ft_cot\
    --load_model_dir gpt2_commonsense_qa_ft_cot_nsm_sigma_0.01/epoch_epoch=4 --devices 0 --batch_size 2

python compute_hessian_traces.py --dataset_key commonsense_qa --model_key gpt2 --train_key ft_cot --preset_key ft_cot\
    --load_model_dir gpt2_commonsense_qa_ft_cot_nsm_sigma_0.012/epoch_epoch=3 --devices 0 --batch_size 2

python compute_hessian_traces.py --dataset_key commonsense_qa --model_key gpt2 --train_key ft_cot --preset_key ft_cot\
    --load_model_dir gpt2_commonsense_qa_ft_cot_nsm_sigma_0.008/epoch_epoch=4 --devices 0 --batch_size 2
