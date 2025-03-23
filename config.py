import torch

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
D = 256
seg_weights = './models/seg.pth'
det_weights = './models/det.pt'
