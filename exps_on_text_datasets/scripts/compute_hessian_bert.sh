for epoch in 1 2 3 4 5 6
do
python compute_hessian_traces.py --config configs/config_glue.json --task_name mrpc --model_name_or_path bert-base-uncased --device 2\
    --checkpoint_dir mrpc_nsm_0.0_0.01_2_False_run_0 --checkpoint_name "model_epoch_${epoch}" --sample_size 20
done
