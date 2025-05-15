nohup python cGAN.py \
  --dataroot datasets/nips \
  --name mae\
  --model cGAN \
  --gpu_ids 0 \
  --niter 50 \
  --save_epoch_freq 25 \
  --lambda_A 10 \
  --lambda_B 10 \
  --adv_weight 100 \
  --output_nc 1 \
  --input_nc 1 \
  --dataset_mode unaligned_mat \
  --training \
  --checkpoints_dir checkpoints/nips \
  --display_single_pane_ncols 1 \
  --mae_finetune \
  --mae_checkpoint_path ../MAPSeg2d/DIR/PROJ_256/Nips_MAE_B_256_new/model_final.pth \
  > logs/nips/train_mae.log 2>&1 &
  # --mae_freeze \