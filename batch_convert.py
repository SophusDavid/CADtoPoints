import numpy as np
from pathlib import Path
import json
import argparse
from utils.convert_utils import calculate_fov_x
def parse_args():
    parser = argparse.ArgumentParser(description='DeepCAD Batch Convert')
    parser.add_argument('--input', type=str, required=True, help='Input directory')
    # parser.add_argument('--output', type=str, required=True, help='Output directory')
    parser.add_argument('--fov_y_degrees', type=int,default=45,help='fov_y_degrees')
    parser.add_argument('--data_type', type=str, default='cad', help='data_type')
    parser.add_argument('--aspect_ratio', type=float, default=4/3, help='aspect_ratio')
    return parser.parse_args()

def convert_cad(args):
    inputdir=Path(args.input)
    # outputdir=Path(args.output)
    fov_y_degrees = args.fov_y_degrees
    fov_x_degrees = calculate_fov_x(fov_y_degrees, args.aspect_ratio)
    transforms={}
    transforms['camera_angle_x']=fov_x_degrees
    frames=[]
    for img in inputdir.iterdir():
        if img.suffix=='.png':
            frame_id=int(img.stem)
        with open(inputdir/'params.json') as f:
            params=json.load(f)
            eye=params[int(img.stem)]["eye"]
            center=params[int(img.stem)]["center"]
            up=params[int(img.stem)]["up"]
            sideright=params[int(img.stem)]["sideright"]
            dir=params[int(img.stem)]["dir"]
            scale=params[int(img.stem)]["scale"]
        trans_matrix=np.linalg.inv(np.array(params[int(img.stem)]["orientation_matrix"]))
        frames.append({"file_path": str(img.stem),
            "rotation": 0.031415926535897934,
            "transform_matrix":trans_matrix.tolist()
            })
        if len(frames)>=300:
            break
    transforms['frames']=frames
    with open(inputdir/'transforms.json', 'w') as f:
        json.dump(transforms, f, indent=4)
def main():
    args = parse_args()
    if args.data_type == 'cad':
        convert_cad(args)
    