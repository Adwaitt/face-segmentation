import os
import gdown

import config as cfg

def get_models():
    if os.path.exists('./models') == False:
        os.mkdir('./models')
        gdown.download(f"https://drive.google.com/uc?id=1J4TC7psd3g2gkQ9edyDrSoNqoLeQ3E4t", cfg.seg_weights, quiet = False)
        gdown.download(f"https://drive.google.com/uc?id=1ni6W-aq3_3U3Qahd5i_m66LoHwtE37k2", cfg.det_weights, quiet = False)
    else:
        print("Files are already downloaded")