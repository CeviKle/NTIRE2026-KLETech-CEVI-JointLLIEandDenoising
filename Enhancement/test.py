import numpy as np
import os
import argparse
from tqdm import tqdm
import cv2

import torch
import torch.nn.functional as F
from glob import glob
from natsort import natsorted
from skimage import img_as_ubyte

import utils
from basicsr.models import create_model
from basicsr.utils.options import parse

############################################
# Arguments
############################################

parser = argparse.ArgumentParser()

parser.add_argument('--opt', type=str, required=True)
parser.add_argument('--weights', type=str, required=True)
parser.add_argument('--dataset', type=str, default='NTIRE')

args = parser.parse_args()

############################################
# Device
############################################

device = torch.device("cpu")

############################################
# Load config
############################################

opt = parse(args.opt, is_train=False)
opt['dist'] = False

input_dir = opt['datasets']['val']['dataroot_lq']

print("Input folder:", input_dir)

############################################
# Load model
############################################

model = create_model(opt).net_g

checkpoint = torch.load(args.weights, map_location=device)

try:
    model.load_state_dict(checkpoint['params'])
except:
    new_checkpoint = {}
    for k in checkpoint['params']:
        new_checkpoint['module.' + k] = checkpoint['params'][k]
    model.load_state_dict(new_checkpoint)

model = model.to(device)
model.eval()

print("Loaded weights:", args.weights)

############################################
# Image list
############################################

input_paths = natsorted(
    glob(os.path.join(input_dir, '*.png')) +
    glob(os.path.join(input_dir, '*.jpg')) +
    glob(os.path.join(input_dir, '*.JPG')) +
    glob(os.path.join(input_dir, '*.PNG'))
)

print("Number of images:", len(input_paths))

############################################
# Output folder
############################################

config = os.path.basename(args.opt).split('.')[0]
checkpoint_name = os.path.basename(args.weights).split('.')[0]

result_dir = os.path.join("results", args.dataset, config, checkpoint_name)
os.makedirs(result_dir, exist_ok=True)

############################################
# Inference
############################################

factor = 4

with torch.no_grad():

    for inp_path in tqdm(input_paths):

        img = np.float32(utils.load_img(inp_path)) / 255.

        img = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).to(device)

        b,c,h,w = img.shape

        H = ((h + factor) // factor) * factor
        W = ((w + factor) // factor) * factor

        padh = H-h if h%factor!=0 else 0
        padw = W-w if w%factor!=0 else 0

        img = F.pad(img,(0,padw,0,padh),'reflect')

        restored = model(img)

        restored = restored[:,:, :h, :w]

        restored = torch.clamp(restored,0,1).cpu().numpy()[0].transpose(1,2,0)

        filename = os.path.basename(inp_path)

        save_path = os.path.join(result_dir, filename)

        output_img = cv2.cvtColor(img_as_ubyte(restored), cv2.COLOR_RGB2BGR)

        cv2.imwrite(
            save_path,
            output_img,
            [int(cv2.IMWRITE_JPEG_QUALITY), 95]
        )

print("Inference complete")
print("Results saved in:", result_dir)