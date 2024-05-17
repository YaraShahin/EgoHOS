from mmseg.apis import inference_segmentor, init_segmentor
import mmcv
import glob
import os
from tqdm import tqdm
import argparse
from PIL import Image
import numpy as np 
from skimage.io import imsave
import pdb, time
import torch
import copy
import cv2

os.makedirs("/home/yara/camera_ws/src/EgoHOS/testimages/pred_twohands/", exist_ok = True)
os.makedirs("/home/yara/camera_ws/src/EgoHOS/testimages/pred_cb/", exist_ok = True)
os.makedirs("/home/yara/camera_ws/src/EgoHOS/testimages/pred_obj1/", exist_ok = True)

curr_time = time.time()
print("[EGOHOS] Initializing models...")

# build the model from a config file and a checkpoint file
model_twohands = init_segmentor("/home/yara/camera_ws/src/EgoHOS/mmsegmentation/work_dirs/seg_twohands_ccda/seg_twohands_ccda.py", "/home/yara/camera_ws/src/EgoHOS/mmsegmentation/work_dirs/seg_twohands_ccda/best_mIoU_iter_56000.pth", device='cuda:0')
model_cb = init_segmentor("/home/yara/camera_ws/src/EgoHOS/mmsegmentation/work_dirs/twohands_to_cb_ccda/twohands_to_cb_ccda.py", "/home/yara/camera_ws/src/EgoHOS/mmsegmentation/work_dirs/twohands_to_cb_ccda/best_mIoU_iter_76000.pth", device='cuda:0')
model_obj = init_segmentor("/home/yara/camera_ws/src/EgoHOS/mmsegmentation/work_dirs/twohands_cb_to_obj1_ccda/twohands_cb_to_obj1_ccda.py","/home/yara/camera_ws/src/EgoHOS/mmsegmentation/work_dirs/twohands_cb_to_obj1_ccda/best_mIoU_iter_34000.pth", device='cuda:0')

print("[EGOHOS] Done; Init Time:", time.time() - curr_time)

alpha = 0.5
device = 'cuda:0'

og_img = "/home/yara/camera_ws/src/EgoHOS/testimages/images/color.png"
hands_img = "/home/yara/camera_ws/src/EgoHOS/testimages/pred_twohands/color.png"
cb_img = "/home/yara/camera_ws/src/EgoHOS/testimages/pred_cb/color.png"
obj_img = "/home/yara/camera_ws/src/EgoHOS/testimages/pred_obj1/workspace_mask.png"

kernel = np.ones((10,10), np.uint8)

while (True):
    if (os.path.exists(og_img)):
        print("[EGOHOS] Segmenting...")
        curr_time = time.time()

        # Making an instance of the built models
        model_twohands_dup = copy.deepcopy(model_twohands)
        model_cb_dup = copy.deepcopy(model_cb)
        model_obj_dup = copy.deepcopy(model_obj)

        model_twohands_dup.to(device)
        model_twohands_dup.eval()

        model_cb_dup.to(device)
        model_cb_dup.eval()

        model_obj_dup.to(device)
        model_obj_dup.eval()
        
        # Segmenting the hand (prequisite to obj)
        seg_result = inference_segmentor(model_twohands_dup, og_img)[0]
        imsave(hands_img, seg_result.astype(np.uint8))
        
        # Segmentig the contact boundary (prequisite to obj)
        seg_result = inference_segmentor(model_cb_dup, og_img)[0]
        imsave(cb_img, seg_result.astype(np.uint8))

        # Segmenting the object
        seg_result = inference_segmentor(model_obj_dup, og_img)[0]
        img = Image.fromarray(seg_result.astype(bool))

        img.save(obj_img,bits=1,optimize=True)

        # Deleting the image as the execution is done
        os.remove(og_img)
        
        # Removing the models from GPU to free space for next model
        del model_twohands_dup
        del model_cb_dup
        del model_obj_dup
        torch.cuda.empty_cache()

        print("[EGOHOS] Finally Done; Time:", time.time() - curr_time)
    else:
        time.sleep(2)


