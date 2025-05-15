nohup python cGAN.py \
  --dataroot datasets/nips \
  --name mae \
  --phase test \
  --output_nc 1 \
  --input_nc 1 \
  --how_many 1 \
  --results_dir results/nips/ \
  --checkpoints_dir checkpoints/nips/ \
  --gpu_ids 1 \
  --mae_finetune \
  --mae_checkpoint_path ../MAPSeg2d/DIR/PROJ_256/Nips_MAE_B_256_new/model_final.pth \
  > logs/nips/test_mae.log 2>&1 &