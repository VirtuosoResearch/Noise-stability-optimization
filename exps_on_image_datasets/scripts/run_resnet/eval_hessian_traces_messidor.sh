python compute_hessian_traces.py --config configs/config_constraint_messidor.json \
    --model ResNet34 --batch_size 16 \
    --checkpoint_dir ResNet34_MessidorDataLoader_none_none_1.0000_1.0000_run_0_data_frac_1.0 \
    --checkpoint_name model_best \
    --sample_size 1000 --device 0 --use_test

python compute_hessian_traces.py --config configs/config_constraint_messidor.json \
    --model ResNet34 --batch_size 16 \
    --checkpoint_dir ResNet34_MessidorDataLoader_ls_0.2 \
    --checkpoint_name model_best \
    --sample_size 1000 --device 0 --use_test

python compute_hessian_traces.py --config configs/config_constraint_messidor.json \
    --model ResNet34 --batch_size 16 \
    --checkpoint_dir ResNet34_MessidorDataLoader_sam_0.01 \
    --checkpoint_name model_best \
    --sample_size 1000 --device 0 --use_test

python compute_hessian_traces.py --config configs/config_constraint_messidor.json \
    --model ResNet34 --batch_size 16 \
    --checkpoint_dir ResNet34_MessidorDataLoader_sam_0.1_ada_True \
    --checkpoint_name model_best \
    --sample_size 1000 --device 0 --use_test

python compute_hessian_traces.py --config configs/config_constraint_messidor.json \
    --model ResNet34 --batch_size 16 \
    --checkpoint_dir ResNet34_MessidorDataLoader_nsm_0.0_0.01_1_True \
    --checkpoint_name model_best \
    --sample_size 1000 --device 0 --use_test

python compute_hessian_traces.py --config configs/config_constraint_messidor.json \
    --model ResNet34 --batch_size 16 \
    --checkpoint_dir ResNet34_MessidorDataLoader_nsm_0.0_0.012_1_True \
    --checkpoint_name model_best \
    --sample_size 1000 --device 0 --use_test

python compute_hessian_traces.py --config configs/config_constraint_messidor.json \
    --model ResNet34 --batch_size 16 \
    --checkpoint_dir ResNet34_MessidorDataLoader_nsm_0.0_0.014_1_True \
    --checkpoint_name model_best \
    --sample_size 1000 --device 0 --use_test


# for seed in 1 2 3
# do
# python compute_hessian_traces.py --config configs/config_constraint_messidor.json \
#     --model ResNet34 --batch_size 16 \
#     --checkpoint_dir ResNet34_MessidorDataLoader_none_none_1.0000_1.0000_run_0 \
#     --checkpoint_name model_epoch_30 \
#     --sample_size 1000 --device 0--seed $seed --use_test
# done

# for seed in 1 2 3
# do
# python compute_hessian_traces.py --config configs/config_constraint_messidor.json \
#     --model ResNet34 --batch_size 16 \
#     --checkpoint_dir ResNet34_MessidorDataLoader_sam_0.01 \
#     --checkpoint_name model_best \
#     --sample_size 1000 --device 0--seed $seed --use_test
# done

# for seed in 1 2 3
# do
# python compute_hessian_traces.py --config configs/config_constraint_messidor.json \
#     --model ResNet34 --batch_size 16 \
#     --checkpoint_dir ResNet34_MessidorDataLoader_nsm_0.0_0.01_1_True \
#     --checkpoint_name model_best \
#     --sample_size 1000 --device 0--seed $seed --use_test
# do

# for seed in 1 2 3
# do
# python compute_hessian_traces.py --config configs/config_constraint_messidor.json \
#     --model ResNet34 --batch_size 16 \
#     --checkpoint_dir ResNet34_MessidorDataLoader_nsm_0.0_0.01_2_True \
#     --checkpoint_name model_best \
#     --sample_size 1000 --device 0--seed $seed --use_test
# do

# for seed in 1 2 3
# python compute_hessian_traces.py --config configs/config_constraint_messidor.json \
#     --model ResNet34 --batch_size 16 \
#     --checkpoint_dir ResNet34_MessidorDataLoader_nsm_0.0_0.01_3_True \
#     --checkpoint_name model_best \
#     --sample_size 1000 --device 0--seed $seed --use_test
# do