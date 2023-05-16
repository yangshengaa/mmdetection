#!/bin/bash
#SBATCH -c 8               
#SBATCH -t 3-00:00           
#SBATCH -p kempner   
#SBATCH --gres=gpu:4
#SBATCH --constraint="a100" 
#SBATCH --mem=160000          
#SBATCH -o out/deepfashion2_%j.out   
#SBATCH -e out/deepfashion2_%j.err 
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=xiaohanzhao@fas.harvard.edu


# setup parameters
# * select model name 
# model_name="queryinst_r101_fpn_2x_coco"
model_name="mask2former_r101_1x"

# * select final epoch for prediction (max 12 for 1x, max 24 for 2x)
epoch=12

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
# python tools/test.py ${CONFIG_FILE} ${work_dir}/epoch_${epoch}.pth --work-dir $work_dir
