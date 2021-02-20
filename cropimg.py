from fastai.learner import load_learner
import cv2
import os
import random
from PIL import Image
import numpy as np
from matplotlib import image
import matplotlib.pyplot as plt

def mask_from_image(fname): 
    fname = str(fname).replace('images','labels').replace('.jpg', "_train_id.png")
    
    return fname

def acc(input, target):
    target = target.squeeze(1)
    mask = target != 20
    return (input.argmax(dim=1)[mask] == target[mask]).float().mean()


def highest_pixels(img, threshold=0):
    highest = []
    for i in range(img.shape[1]):
        column = img[:, i]
        wanted = np.where(column > threshold)[0]
        if len(wanted) == 0:
            highest.append(0)
        else:
            highest.append(wanted[0])
            
    return highest

def crop_black_top(img, highest_road_pixels, height):
    cutoff_at = min(np.min(highest_road_pixels), int(height / 2.5))
    height_multiplyer = img.shape[0] / height
    cutoff_at = int(cutoff_at * height_multiplyer)

    return img[cutoff_at:, :, :]

def crop_top(img):
    highest_road_pixels = [p for p in highest_pixels(img) if p > 0]

    if len(highest_road_pixels) == 0:
        return img

    return crop_black_top(img, highest_road_pixels, img.shape[0])


def expand_mask(binary_mask):
    mask = cv2.dilate(np.invert(cv2.medianBlur(binary_mask, 103)), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30,30)))
    
    
    if np.sum(mask) < np.prod(mask.shape) / 3 * 255:
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (140,140)))
        
    mask = (mask / 255).astype("uint8")
    
    return mask
    
def hide_non_road(img, mask):
    return np.expand_dims(cv2.resize(mask, dsize=img.shape[:2]), 2) * img

def road_preds(preds):
    classes = preds[0]
    
    classes = np.where(classes==20, 0, classes)
    classes = np.where(classes==1, 0, classes)
    classes = np.where(classes==13, 0, classes)
    classes = np.where(classes==14, 0, classes)
    classes = np.where(classes==15, 0, classes)
    classes = np.where(classes==17, 0, classes)
    
    return (np.where(classes>0, 1, classes) * 255).astype("uint8")

class UnetModelWrapper():
    def __init__(self, model_path) -> None:
        self.loaded = False 
        self.model_path = model_path
        self.model = None
    
    def __call__(self, img):
        if not self.loaded:
            self.model = load_learner(self.model_path, cpu=False)
            self.loaded = True
        
        original_size = (img.shape[1], img.shape[0])

        mask = expand_mask(road_preds(self.model.predict(img)))
        masked_img = hide_non_road(img, mask)
        return masked_img

def full_contents(path):
    return [path + f for f in os.listdir(path)]

def get_imgs_r(directory, suffixes=[".jpg", ".jpeg"]):
    imgs = []
    
    if not os.path.isdir(directory):
        if any([s in directory for s in suffixes]):
            return [directory]
        return []
    
    directory += "/"
    
    children = full_contents(directory)
    
    for child in children:
        imgs.extend(get_imgs_r(child))
    
    return imgs

all_imgs = []

img_folders = [
    "/home/lewington/code/rddc2020/yolov5/datasets/road2020/test1",
    "/home/lewington/code/rddc2020/yolov5/datasets/road2020/test2"
]

for folder in img_folders:
    all_imgs.extend(get_imgs_r(folder))
    
unet = UnetModelWrapper("/home/lewington/code/faultnet/app/unet/models/frozen-15epochs")

for path in all_imgs:
    data = image.imread(path)
    cropped = unet(data)    
    
    Image.fromarray(cropped).save(path)
    print(path)
