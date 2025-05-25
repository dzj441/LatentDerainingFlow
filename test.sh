accelerate launch \
  --num_machines=1 --num_processes=1 --machine_rank=0 --gpu_ids=1\
  test_scripts/test_latentderain.py \
    --ckpt_path="./latent_derain_checkpoints/SD2vae_checkpoints/checkpoint.100000.pt" \
    --input_dir="./datasets/LHP/test/input" \
    --output_dir="./test_on_datasets/SD2LHP_Unet16_ss50" \
    --image_size=256 \
    --vae_model="sd2-vae" \
    --sampling_steps=50\

accelerate launch \
  --num_machines=1 --num_processes=1 --machine_rank=0 --gpu_ids=1\
  test_scripts/test_latentderain.py \
    --ckpt_path="./latent_derain_checkpoints/SD2vae_checkpoints/checkpoint.100000.pt" \
    --input_dir="./datasets/RealRain1k/RealRain-1k/RealRain-1k-H/test/input" \
    --output_dir="./test_on_datasets/SD2RealRain_Unet16_ss50" \
    --image_size=256 \
    --vae_model="sd2-vae" \
    --sampling_steps=50\

accelerate launch \
  --num_machines=1 --num_processes=1 --machine_rank=0 --gpu_ids=1\
  test_scripts/test_latentderain.py \
    --ckpt_path="./latent_derain_checkpoints/SD2Unet32_vae_checkpoints/checkpoint.100000.pt" \
    --input_dir="./datasets/LHP/test/input" \
    --output_dir="./test_on_datasets/SD2LHP_Unet32_ss50" \
    --image_size=256 \
    --vae_model="sd2-vae" \
    --sampling_steps=50\

accelerate launch \
  --num_machines=1 --num_processes=1 --machine_rank=0 --gpu_ids=1\
  test_scripts/test_latentderain.py \
    --ckpt_path="./latent_derain_checkpoints/SD2Unet32_vae_checkpoints/checkpoint.100000.pt" \
    --input_dir="./datasets/RealRain1k/RealRain-1k/RealRain-1k-H/test/input" \
    --output_dir="./test_on_datasets/SD2RealRain_Unet32_ss50" \
    --image_size=256 \
    --vae_model="sd2-vae" \
    --sampling_steps=50\