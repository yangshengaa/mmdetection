"""
copy configs from deepfashion2 and modify test pipeline
"""

import os 
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--model", default=None, type=str, help='the model configs to copy over')

args = parser.parse_args()

if __name__ == '__main__':
    with open(f"configs/deepfashion2/{args.model}.py", 'r') as f:
        content = f.read()

    for cat in ["shop", "user", "zoom_in_1", "zoom_in_2", "zoom_in_3", "viewpoint_1", "viewpoint_2", "viewpoint_3", "scale_1", "scale_2", "scale_3", "occlusion_1", "occlusion_2", "occlusion_3"]:
        new_content = content.replace("data_pipeline.py", f"data_pipeline_{cat}.py")
        with open(f"configs/deepfashion2_test_cat/{args.model}_{cat}.py", 'w') as f:
            f.write(new_content)
