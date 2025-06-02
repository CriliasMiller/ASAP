# EXPID='your_best_model_dir_name'
# EXPID='log20240403_164916'
EXPID='log20250412_181657_train_without_GCN'
export CUDA_VISIBLE_DEVICES=1
HOST='127.0.0.1'
PORT='1'

NUM_GPU=1

python test.py \
--config 'configs/test.yaml' \
--output_dir 'results' \
--launcher pytorch \
--rank 0 \
--log_num ${EXPID} \
--dist-url tcp://${HOST}:1003${PORT} \
--token_momentum \
--world_size $NUM_GPU \
--test_epoch best \

