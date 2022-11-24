
import matplotlib.pyplot as plt
from scpye.detect.fruit_detector import FruitDetector
from scpye.improc.binary_cleaner import BinaryCleaner
from scpye.utils.data_manager import DataManager
from scpye.utils.drawing import imshow
from scpye.improc.image_processing import (enhance_contrast, gray_from_bgr,
                                           bgr_from_gray, swap_channels)
import numpy as np

# %%
base_dir = '/home/chao/Workspace/dataset/agriculture'
color = 'red'
mode = 'slow_flash'
side = 'north'

# %%
dm = DataManager(base_dir, color=color, mode=mode, side=side)
fd = FruitDetector.from_pickle(dm)
bc = BinaryCleaner(ksize=5, iters=1)

I, L = dm.load_image_and_label(8)
It, Lt, bw = fd.detect_image_label(I, L)
# Clean binary mask here
bw = bc.clean(bw)

disp_bw = np.zeros_like(bw)
disp_bw[(Lt & bw) > 0] = 2
disp_bw[(Lt ^ bw) > 0] = 1

disp_bgr = enhance_contrast(It)
disp = bgr_from_gray(gray_from_bgr(disp_bgr))

disp[bw > 0] = disp_bgr[bw > 0]
disp = swap_channels(disp)

f, ax = imshow(disp, disp_bw, figsize=(15, 18), cmap=plt.cm.viridis,
               interp='none', hide_axes=True)
