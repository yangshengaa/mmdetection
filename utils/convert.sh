#!/bin/bash
#SBATCH -c 4               
#SBATCH -t 1-00:00           
#SBATCH -p shared 
#SBATCH --mem=16000  
#SBATCH -o convert_%j.out   
#SBATCH -e convert_%j.err 
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=shengyang@fas.harvard.edu

# specify path
path="/n/pehlevan_lab/Everyone/deepfashion2/raw/test"

# run convertion
# python deepfashion2_to_coco.py --path $path
python partition_dataset_image.py --path $path
python partition_dataset_item.py --path $path