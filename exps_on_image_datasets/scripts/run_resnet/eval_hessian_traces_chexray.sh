python compute_hessian_traces.py --config configs/config_constraint_chexray.json \
    --model ResNet34 --batch_size 16 \
    --checkpoint_dir ResNet34_CXRDataLoader_none_none_1.0000_1.0000_run_0 \
    --checkpoint_name model_best \
    --sample_size 1000 --device 0  --data_frac 0.1 --downsample_test


python compute_hessian_traces.py --config configs/config_constraint_chexray.json \
    --model ResNet34 --batch_size 16 \
    --checkpoint_dir ResNet34_CXRDataLoader_sam_0.01_ada_False \
    --checkpoint_name model_best \
    --sample_size 1000 --device 0  --data_frac 0.1 --downsample_test


python compute_hessian_traces.py --config configs/config_constraint_chexray.json \
    --model ResNet34 --batch_size 16 \
    --checkpoint_dir ResNet34_CXRDataLoader_sam_0.1_ada_True \
    --checkpoint_name model_best \
    --sample_size 1000 --device 0  --data_frac 0.1 --downsample_test
    

python compute_hessian_traces.py --config configs/config_constraint_chexray.json \
    --model ResNet34 --batch_size 16 \
    --checkpoint_dir ResNet34_CXRDataLoader_nsm_0.0_0.01_1_True \
    --checkpoint_name model_best \
    --sample_size 1000 --device 0  --data_frac 0.1 --downsample_test


python compute_hessian_traces.py --config configs/config_constraint_chexray.json \
    --model ResNet34 --batch_size 16 \
    --checkpoint_dir ResNet34_CXRDataLoader_nsm_0.0_0.012_1_True \
    --checkpoint_name model_best \
    --sample_size 1000 --device 0  --data_frac 0.1 --downsample_test