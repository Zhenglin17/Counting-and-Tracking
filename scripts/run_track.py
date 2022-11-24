import logging
from tqdm import tqdm
from scpye.improc.binary_cleaner import BinaryCleaner
from scpye.improc.blob_analyzer import BlobAnalyzer
from scpye.track.fruit_tracker import FruitTracker
from scpye.utils.data_manager import DataManager
from scpye.utils.bag_manager import BagManager
from scpye.utils.fruit_visualizer import FruitVisualizer

#TODO: my idea: load our segmented images from ?? , load our bboxes in create_data.py file,

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
ft = FruitTracker()
fv = FruitVisualizer(pause_time=0.1)

# %%
#TODO: refer to image_processing.py and blob_analyzer: bw is greyscale image, bgr here is color image, fruits are just a list of bboxes and bw is black and white image with bounding boxes in it(bw: binary image)
for bgr, bw in tqdm(bm.load_detect()):
    # TODO: what are the following two steps for?????????????????????????what is the final bw we imput to the ft.track???
    bw = bc.clean(bw)
    # fruits, bw = ba.analyze(bgr, bw)
    ft.track(bgr, fruits, bw)
#    fv.show(ft.disp_bgr, ft.disp_bw)
    bm.save_track(ft.disp_bgr, ft.disp_bw, save_disp=True)

ft.finish()
