for seed in 0 1 2
do
python compute_hessian_traces.py --config configs/config_constraint_aircrafts.json \
    --model ResNet34 --batch_size 16 \
    --checkpoint_dir ResNet34_AircraftsDataLoader_bsam_0.01_0.01 \
    --checkpoint_name model_best \
    --sample_size 1000 --device 1 --seed $seed

python compute_hessian_traces.py --config configs/config_constraint_indoor.json \
    --model ResNet34 --batch_size 16 \
    --checkpoint_dir ResNet34_IndoorDataLoader_bsam_0.01_0.01 \
    --checkpoint_name model_best \
    --sample_size 1000 --device 1 --seed $seed

python compute_hessian_traces.py --config configs/config_constraint_messidor.json \
    --model ResNet34 --batch_size 16 \
    --checkpoint_dir ResNet34_MessidorDataLoader_bsam_0.01_0.01 \
    --checkpoint_name model_best \
    --sample_size 1000 --device 1 --seed $seed

python compute_hessian_traces.py --config configs/config_constraint_aptos.json \
    --model ResNet34 --batch_size 16 \
    --checkpoint_dir ResNet34_AptosDataLoader_bsam_0.01_0.01 \
    --checkpoint_name model_best \
    --sample_size 1000 --device 1 --seed $seed
done