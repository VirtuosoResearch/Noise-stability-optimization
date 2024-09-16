# python compute_noise_stability.py --config configs/config_glue.json --task_name mrpc --model_name_or_path bert-base-uncased --device 1 \
# --checkpoint_dir mrpc_False_none_0_noise_rate_0.0_run_0 --checkpoint_name model_best --sample_size 100 --eps 0.01 --compute_hessian_trace

# 0.01 0.02 0.03 0.04 0.05 0.06 0.07
# for eps in 0.0071 0.0072 0.0073 0.0074 0.0075 0.0076 0.0077 0.0078 0.0079 0.0080
# do
# python compute_noise_stability.py --config configs/config_glue.json --task_name mrpc --model_name_or_path bert-base-uncased --device 1 \
# --checkpoint_dir mrpc_False_none_0_noise_rate_0.0_run_0 --checkpoint_name model_best --sample_size 100 --eps $eps
# done

# for epoch in 1 2 3 4 5 6 7 8 9 10
# do
# python compute_noise_stability.py --config configs/config_glue.json --task_name mrpc --model_name_or_path bert-base-uncased --device 3 \
# --checkpoint_dir mrpc_False_none_0_noise_rate_0.0_run_0 --checkpoint_name model_best --sample_size 100 --eps 0.01 --compute_hessian_trace \
# --epoch $epoch
# done

# python compute_hessian_traces.py --config configs/config_glue.json --task_name mrpc --model_name_or_path bert-base-uncased --device 3 \
# --checkpoint_dir mrpc_False_none_0_noise_rate_0.0_run_0 --checkpoint_name model_best --sample_size 10 \
# --epoch 10

for epoch in 1 2 3 4 5 6 7 8 9 10
do
python compute_noise_stability.py --config configs/config_glue.json --task_name mrpc --model_name_or_path bert-base-uncased --device 2 \
--checkpoint_dir mrpc_nsm_0.0_0.01_2_False_run_0 --checkpoint_name model_best --sample_size 100 --eps 0.01 --compute_hessian_trace \
--epoch $epoch
done

# python compute_hessian_traces.py --config configs/config_glue.json --task_name mrpc --model_name_or_path bert-base-uncased --device 3 \
# --checkpoint_dir mrpc_False_none_0_noise_rate_0.0_run_0 --checkpoint_name model_best --sample_size 10 \
# --epoch 10