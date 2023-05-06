# setup parameters

# * select model name 
model_name="mask-rcnn_r101_fpn_2x"

# * select final epoch for prediction (12 for 1x, 24 for 2x)
epoch=24

# * select saving directory
work_dir="/n/pehlevan_lab/Users/shengy/fashion/work_dir/${model_name}"

CONFIG_FILE="configs/deepfashion2/${model_name}.py"
GPU_NUM=4

# multigpu
bash tools/dist_train.sh \
    ${CONFIG_FILE} \
    ${GPU_NUM} \
    --work-dir $work_dir

# single gpu
# python tools/train.py ${CONFIG_FILE} --work-dir $work_dir

# single gpu test evaluation
python tools/test.py ${CONFIG_FILE} ${work_dir}/epoch_${epoch}.pth
