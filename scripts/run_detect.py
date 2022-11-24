import logging
from tqdm import tqdm
from scpye.detect.fruit_detector import FruitDetector
from scpye.utils.data_manager import DataManager
from scpye.utils.bag_manager import BagManager
from scpye.utils.fruit_visualizer import FruitVisualizer

logging.basicConfig(level=logging.INFO)

base_dir = '/home/chao/Workspace/dataset/agriculture'
color = 'red'
mode = 'slow_flash'
side = 'north'
bag_ind = 4

# %%
dm = DataManager(base_dir, color=color, mode=mode, side=side)
bm = BagManager(dm.data_dir, bag_ind)
fd = FruitDetector.from_pickle(dm)
fv = FruitVisualizer()

# %%
for image in tqdm(bm.load_bag()):
    bgr, bw = fd.detect(image)
#    fv.show(bgr, bw)
    bm.save_detect(bgr, bw)
