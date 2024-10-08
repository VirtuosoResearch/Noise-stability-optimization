# python compute_hessian_traces.py --config configs/config_constraint_aircrafts.json \
#     --model ResNet34 --batch_size 4 \
#     --checkpoint_dir ResNet34_AircraftsDataLoader_ls_0.2 \
#     --checkpoint_name model_best \
#     --sample_size 1000 --device 3

# python compute_hessian_traces.py --config configs/config_constraint_aircrafts.json \
#     --model ResNet34 --batch_size 4 \
#     --checkpoint_dir ResNet34_AircraftsDataLoader_ls_0.4 \
#     --checkpoint_name model_best \
#     --sample_size 1000 --device 3

# python compute_hessian_traces.py --config configs/config_constraint_aircrafts.json \
#     --model ResNet34 --batch_size 4 \
#     --checkpoint_dir ResNet34_AircraftsDataLoader_swa_20_0.0002 \
#     --load_multiple_points --checkpoint_names model_epoch_20 model_epoch_25 model_epoch_30 \
#     --sample_size 1000 --device 3

# for seed in 0 1 2
# do
# python compute_hessian_traces.py --config configs/config_constraint_aircrafts.json \
#     --model ResNet34 --batch_size 16 \
#     --checkpoint_dir ResNet34_AircraftsDataLoader_nsm_0.0_0.008_1_True_swa_20_0.0002 \
#     --load_multiple_points --checkpoint_names model_epoch_20 model_epoch_21 model_epoch_22 model_epoch_23 model_epoch_24 model_epoch_25 model_epoch_26 model_epoch_27 model_epoch_28 model_epoch_29 model_epoch_30 \
#     --sample_size 1000 --device 0 --seed $seed

# python compute_hessian_traces.py --config configs/config_constraint_aircrafts.json \
#     --model ResNet34 --batch_size 16 \
#     --checkpoint_dir ResNet34_AircraftsDataLoader_swa_20_0.0002 \
#     --load_multiple_points --checkpoint_names model_epoch_20 model_epoch_21 model_epoch_22 model_epoch_23 model_epoch_24 model_epoch_25 model_epoch_26 model_epoch_27 model_epoch_28 model_epoch_29 model_epoch_30 \
#     --sample_size 1000 --device 0 --seed $seed
# done

python compute_hessian_traces.py --config configs/config_constraint_aircrafts.json \
    --model ResNet34 --batch_size 16 \
    --checkpoint_dir ResNet34_AircraftsDataLoader_nsm_1.0_0.01_2_False \
    --checkpoint_name model_best \
    --sample_size 1000 --device 2 --seed 0

python compute_hessian_traces.py --config configs/config_constraint_aircrafts.json \
    --model ResNet34 --batch_size 16 \
    --checkpoint_dir ResNet34_AircraftsDataLoader_nsm_1.0_0.01_4_False \
    --checkpoint_name model_best \
    --sample_size 1000 --device 2 --seed 0

python compute_hessian_traces.py --config configs/config_constraint_aircrafts.json \
    --model ResNet34 --batch_size 16 \
    --checkpoint_dir ResNet34_AircraftsDataLoader_nsm_1.0_0.01_6_False \
    --checkpoint_name model_best \
    --sample_size 1000 --device 2 --seed 0