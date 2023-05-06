# setup parameters
work_dir="/n/pehlevan_lab/Users/shengy/fashion/work_dir/mask-rcnn_r50_fpn_1x"

CONFIG_FILE="configs/deepfashion2/mask-rcnn_r50_fpn_1x.py"
GPU_NUM=4

# multigpu
bash tools/dist_train.sh \
    ${CONFIG_FILE} \
    ${GPU_NUM} \
    --work-dir $work_dir

# single gpu
# python tools/train.py ${CONFIG_FILE} --work-dir $work_dir

# single gpu test evaluation
python tools/test.py ${CONFIG_FILE} ${work_dir}/epoch_12.pth