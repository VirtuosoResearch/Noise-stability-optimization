for seed in 0 1 2
do
python compute_hessian_traces.py --config configs/config_constraint_aircrafts.json \
    --model ResNet34 --batch_size 16 \
    --checkpoint_dir ResNet34_AircraftsDataLoader_rsam_0.01_0.01_1.0 \
    --checkpoint_name model_best \
    --sample_size 1000 --device 3 --seed $seed

python compute_hessian_traces.py --config configs/config_constraint_indoor.json \
    --model ResNet34 --batch_size 16 \
    --checkpoint_dir ResNet34_IndoorDataLoader_rsam_0.01_0.01_1.0 \
    --checkpoint_name model_best \
    --sample_size 1000 --device 3 --seed $seed

python compute_hessian_traces.py --config configs/config_constraint_messidor.json \
    --model ResNet34 --batch_size 16 \
    --checkpoint_dir ResNet34_MessidorDataLoader_rsam_0.01_0.01_1.0 \
    --checkpoint_name model_best \
    --sample_size 1000 --device 3 --seed $seed

python compute_hessian_traces.py --config configs/config_constraint_aptos.json \
    --model ResNet34 --batch_size 16 \
    --checkpoint_dir ResNet34_AptosDataLoader_rsam_0.01_0.01_1.0 \
    --checkpoint_name model_best \
    --sample_size 1000 --device 3 --seed $seed
done