# python compute_hessian_traces.py --config configs/config_constraint_aptos.json \
#     --model ResNet34 --batch_size 16 \
#     --checkpoint_dir ResNet34_AptosDataLoader_none_none_1.0000_1.0000_run_0_data_frac_1.0 \
#     --checkpoint_name model_best \
#     --sample_size 1000 --device 0 --use_test

# python compute_hessian_traces.py --config configs/config_constraint_aptos.json \
#     --model ResNet34 --batch_size 16 \
#     --checkpoint_dir ResNet34_AptosDataLoader_ls_0.2 \
#     --checkpoint_name model_best \
#     --sample_size 1000 --device 0 --use_test

# python compute_hessian_traces.py --config configs/config_constraint_aptos.json \
#     --model ResNet34 --batch_size 16 \
#     --checkpoint_dir ResNet34_AptosDataLoader_sam_0.01_ada_False \
#     --checkpoint_name model_best \
#     --sample_size 1000 --device 0 --use_test

python compute_hessian_traces.py --config configs/config_constraint_aptos.json \
    --model ResNet34 --batch_size 16 \
    --checkpoint_dir ResNet34_AptosDataLoader_sam_0.1_ada_True \
    --checkpoint_name model_best \
    --sample_size 1000 --device 3 --use_test

python compute_hessian_traces.py --config configs/config_constraint_aptos.json \
    --model ResNet34 --batch_size 16 \
    --checkpoint_dir ResNet34_AptosDataLoader_nsm_0.0_0.01_1_True \
    --checkpoint_name model_best \
    --sample_size 1000 --device 3 --use_test

python compute_hessian_traces.py --config configs/config_constraint_aptos.json \
    --model ResNet34 --batch_size 16 \
    --checkpoint_dir ResNet34_AptosDataLoader_nsm_0.0_0.012_1_True \
    --checkpoint_name model_best \
    --sample_size 1000 --device 3 --use_test

python compute_hessian_traces.py --config configs/config_constraint_aptos.json \
    --model ResNet34 --batch_size 16 \
    --checkpoint_dir ResNet34_AptosDataLoader_nsm_0.0_0.014_1_True \
    --checkpoint_name model_best \
    --sample_size 1000 --device 3 --use_test
