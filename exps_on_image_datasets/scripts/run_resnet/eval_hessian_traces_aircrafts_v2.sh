for seed in 0 1 2
do
python compute_hessian_traces.py --config configs/config_constraint_aircrafts.json \
    --model ResNet34 --batch_size 16 \
    --checkpoint_dir ResNet34_AircraftsDataLoader_none_none_1.0000_1.0000_run_0 \
    --checkpoint_name model_best \
    --sample_size 1000 --device 1 --seed $seed

python compute_hessian_traces.py --config configs/config_constraint_aircrafts.json \
    --model ResNet34 --batch_size 16 \
    --checkpoint_dir ResNet34_AircraftsDataLoader_nsm_0.0_0.01_1_True \
    --checkpoint_name model_best \
    --sample_size 1000 --device 1 --seed $seed
done