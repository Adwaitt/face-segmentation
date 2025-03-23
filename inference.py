import config as cfg

from skimage import io

import torch

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from load_model import load_model
from postprocess import postprocess

import matplotlib.pyplot as plt

def process_input(image):
    tfms = A.Compose([
    A.Resize(cfg.D, cfg.D),
    A.Normalize(mean = 0.0, std = 1.0, max_pixel_value = 255.0),
    ToTensorV2()
    ])
    img = tfms(image = image)['image']
    return img

def get_output(image):
    seg, det = load_model()
    if image.shape[-1] == 4:
        image = image[..., :3]
    bboxes = det(image, conf = 0.50, verbose = False, imgsz = 320)
    num_bboxes = len(bboxes[0].boxes.data)
    if num_bboxes > 1:
        raise ValueError("More than one face detected! Please select an image that fits the criteria.")
    elif num_bboxes < 1:
        raise ValueError("No face detected! Please select an image that fits the criteria.")
    else:
        seg.eval()
        image = process_input(image)
        img = torch.unsqueeze(image, axis = 0)
        with torch.no_grad():
            y = seg(img.to(cfg.device))
        y = torch.squeeze(y).cpu().numpy()
        mask = y
        out = postprocess(image, mask)
        return out

if __name__ == '__main__':
    image = io.imread("C:/Users/adwai/OneDrive/Desktop/BELKA/images/3.jpg")
    if image.shape[-1] == 4:
        image = image[..., :3]
    face = get_output(image)
    plt.imshow(face)
    plt.show()