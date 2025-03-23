import config as cfg

import torch
from network import Network

from ultralytics import YOLO

def load_model(seg_weights = cfg.seg_weights, det_weights = cfg.det_weights, device = cfg.device):
    seg = Network().to(device)
    seg_weights = torch.load(seg_weights, map_location = device, weights_only = False)
    seg.load_state_dict(seg_weights)
    det = YOLO(det_weights)
    return seg, det

if __name__ == '__main__':
    model = load_model()
                          
