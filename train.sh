nvidia-smi

echo "start training"

WANDB_MODE=offline accelerate launch --gpu_ids=1 --num_machines=1 --num_processes=1 --machine_rank=0 train_scripts/train_latentderain.py \
    --wandb_project="derain" \
