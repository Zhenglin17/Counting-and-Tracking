import cv2
import logging
from scpye.improc.binary_cleaner import BinaryCleaner
from scpye.improc.blob_analyzer import BlobAnalyzer
from scpye.improc.image_processing import enhance_contrast
from scpye.utils.data_manager import DataManager
from scpye.utils.bag_manager import BagManager
from scpye.utils.drawing import draw_blob_analyzer
from scpye.utils.fruit_visualizer import FruitVisualizer

logging.basicConfig(level=logging.INFO)

# %%
base_dir = '/home/chao/Workspace/dataset/agriculture'
color = 'red'
mode = 'slow_flash'
side = 'north'
bag_ind = 1

# %%
dm = DataManager(base_dir, color=color, mode=mode, side=side)
bm = BagManager(dm.data_dir, bag_ind)
bc = BinaryCleaner(ksize=5, iters=1)
ba = BlobAnalyzer()
fv = FruitVisualizer(pause_time=0.5)

# %%
for bgr, bw in bm.load_detect():
    bw = bc.clean(bw)
    fruits, bw = ba.analyze(bgr, bw)

    disp_bgr = enhance_contrast(bgr)
    disp_bw = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)

    draw_blob_analyzer(ba, disp_bgr, disp_bw)

    fv.show(disp_bgr, disp_bw)
