import numpy as np
import cv2

def postprocess(image, mask):
    image = image.permute(1, -1, 0).numpy()
    mask = np.expand_dims(mask, axis = -1)
    mask = (mask > 0.5).astype(np.uint8)
    
    i_x, i_y, _ = image.shape
    m_x, m_y, _ = mask.shape
    
    x_m = min(i_x, m_x)
    x_half_m = mask.shape[0]//2
    
    m_mask = mask[x_half_m - x_m // 2 : x_half_m + x_m // 2+1, :i_y]
    
    i_width_half = image.shape[1]//2
    i_to_mask = image[:, i_width_half - x_half_m: i_width_half + x_half_m]
    masked = cv2.bitwise_and(i_to_mask, i_to_mask, mask = m_mask)

    tmp = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    _,alpha = cv2.threshold(tmp,0,255,cv2.THRESH_BINARY)
    b, g, r = cv2.split(masked)
    rgba = [b,g,r, alpha]
    masked_tr = cv2.merge(rgba,4)
    
    return masked_tr