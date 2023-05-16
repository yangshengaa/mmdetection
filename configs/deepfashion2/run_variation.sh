#!/bin/bash
#SBATCH -c 4               
#SBATCH -t 1-00:00           
#SBATCH -p kempner   
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100" 
#SBATCH --mem=80000          
#SBATCH -o out/deepfashion2_%j.out   
#SBATCH -e out/deepfashion2_%j.err 
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=shengyang@fas.harvard.edu


# setup parameters
# * select model name 
# model_name="mask-rcnn_r50_fpn_2x"
# model_name="queryinst_r101_fpn_2x_coco"
# model_name="queryinst_r50_fpn_2x_coco"
model_name="mask2former_r50_2x"

# * select final epoch for prediction (max 12 for 1x, max 24 for 2x)
epoch=19

# * select saving directory
save_dir="/n/pehlevan_lab/Users/shengy/fashion/performance_by_cat/${model_name}"
work_dir="/n/pehlevan_lab/Users/shengy/fashion/work_dir/${model_name}"
GPU_NUM=1

# ================= modify config scripts ==============
echo "modifying scripts ..."
python configs/deepfashion2_test_cat/config_copy.py --model $model_name

# ================== compute ===================
# shop
echo "shop"
CONFIG_FILE="configs/deepfashion2_test_cat/${model_name}_shop.py"
python tools/test.py ${CONFIG_FILE} ${work_dir}/epoch_${epoch}.pth --work-dir ${save_dir}/shop

# user
echo "user"
CONFIG_FILE="configs/deepfashion2_test_cat/${model_name}_user.py"
python tools/test.py ${CONFIG_FILE} ${work_dir}/epoch_${epoch}.pth --work-dir ${save_dir}/user

# scale_1
echo "scale_1"
CONFIG_FILE="configs/deepfashion2_test_cat/${model_name}_scale_1.py"
python tools/test.py ${CONFIG_FILE} ${work_dir}/epoch_${epoch}.pth --work-dir ${save_dir}/scale_1

# scale_2
echo "scale_2"
CONFIG_FILE="configs/deepfashion2_test_cat/${model_name}_scale_2.py"
python tools/test.py ${CONFIG_FILE} ${work_dir}/epoch_${epoch}.pth --work-dir ${save_dir}/scale_2

# scale_3
echo "scale_3"
CONFIG_FILE="configs/deepfashion2_test_cat/${model_name}_scale_3.py"
python tools/test.py ${CONFIG_FILE} ${work_dir}/epoch_${epoch}.pth --work-dir ${save_dir}/scale_3

# occlusion_1
echo "occlusion_1"
CONFIG_FILE="configs/deepfashion2_test_cat/${model_name}_occlusion_1.py"
python tools/test.py ${CONFIG_FILE} ${work_dir}/epoch_${epoch}.pth --work-dir ${save_dir}/occlusion_1

# occlusion_2
echo "occlusion_2"
CONFIG_FILE="configs/deepfashion2_test_cat/${model_name}_occlusion_2.py"
python tools/test.py ${CONFIG_FILE} ${work_dir}/epoch_${epoch}.pth --work-dir ${save_dir}/occlusion_2

# occlusion_3
echo "occlusion_3"
CONFIG_FILE="configs/deepfashion2_test_cat/${model_name}_occlusion_3.py"
python tools/test.py ${CONFIG_FILE} ${work_dir}/epoch_${epoch}.pth --work-dir ${save_dir}/occlusion_3

# zoom_in_1
echo "zoom_in_1"
CONFIG_FILE="configs/deepfashion2_test_cat/${model_name}_zoom_in_1.py"
python tools/test.py ${CONFIG_FILE} ${work_dir}/epoch_${epoch}.pth --work-dir ${save_dir}/zoom_in_1

# zoom_in_2
echo "zoom_in_2"
CONFIG_FILE="configs/deepfashion2_test_cat/${model_name}_zoom_in_2.py"
python tools/test.py ${CONFIG_FILE} ${work_dir}/epoch_${epoch}.pth --work-dir ${save_dir}/zoom_in_2

# zoom_in_3
echo "zoom_in_3"
CONFIG_FILE="configs/deepfashion2_test_cat/${model_name}_zoom_in_3.py"
python tools/test.py ${CONFIG_FILE} ${work_dir}/epoch_${epoch}.pth --work-dir ${save_dir}/zoom_in_3

# viewpoint_1
echo "viewpoint_1"
CONFIG_FILE="configs/deepfashion2_test_cat/${model_name}_viewpoint_1.py"
python tools/test.py ${CONFIG_FILE} ${work_dir}/epoch_${epoch}.pth --work-dir ${save_dir}/viewpoint_1

# viewpoint_2
echo "viewpoint_2"
CONFIG_FILE="configs/deepfashion2_test_cat/${model_name}_viewpoint_2.py"
python tools/test.py ${CONFIG_FILE} ${work_dir}/epoch_${epoch}.pth --work-dir ${save_dir}/viewpoint_2

# viewpoint_3
echo "viewpoint_3"
CONFIG_FILE="configs/deepfashion2_test_cat/${model_name}_viewpoint_3.py"
python tools/test.py ${CONFIG_FILE} ${work_dir}/epoch_${epoch}.pth --work-dir ${save_dir}/viewpoint_3
