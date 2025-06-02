EXPID=$(date +"%Y%m%d_%H%M%S")_train_without_GCN
# export CUDA_VISIBLE_DEVICES=2
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
HOST='127.0.1.1'
PORT='1'

NUM_GPU=8
python train.py \
--config 'configs/train.yaml' \
--output_dir 'results' \
--checkpoint '/workspace/Crilias/zhangzhenxing/coDE/results/log20250410_115817_Pretrain_without_GCN/checkpoint_best.pth' \
--launcher pytorch \
--rank 0 \
--log_num ${EXPID} \
--dist-url tcp://${HOST}:1001${PORT} \
--token_momentum \
--world_size $NUM_GPU \
--model_save_epoch 100 \
