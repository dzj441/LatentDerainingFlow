

# python3 -u eval.py \
#   --dir1 ./test_on_datasets/LHPgt \
#   --dir2 ./test_on_datasets/SD2Unet16_mse_CFM__LHP \
#   --log_file eval.log


# python3 -u eval.py \
#   --dir1 ./test_on_datasets/RealRaingt \
#   --dir2 ./test_on_datasets/SD2Unet16_mse_CFM_RealRain \
#   --log_file eval.log



python3 -u eval.py \
  --dir1 ./test_on_datasets/SD2Unet32_mse_freq_CFM_30k_hybrid_LHP \
  --dir2 ./test_on_datasets/LHPgt \
  --log_file final.log


# python3 -u eval.py \
#   --dir1 ./test_on_datasets/RealRaingt \
#   --dir2 ./test_on_datasets/SD2Unet32_mse_freq_CFM_30k_hybrid_RealRain \
#   --log_file final.log


# python3 -u eval.py \
#   --dir1 ./test_on_datasets/LHPgt \
#   --dir2 ./test_on_datasets/pixel_Unet32_mse_freq_CFM_30k_hybrid_LHP \
#   --log_file final.log


# python3 -u eval.py \
#   --dir1 ./test_on_datasets/RealRaingt \
#   --dir2 ./test_on_datasets/pixel_Unet32_mse_freq_CFM_30k_hybrid_RealRain \
#   --log_file final.log

