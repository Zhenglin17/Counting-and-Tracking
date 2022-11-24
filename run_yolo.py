import subprocess
import os
def run_yolo():
    subprocess.call(['./darknet/darknet', 'detector', 'test', './darknet/data/obj.data', './darknet/cfg/yolov4-tiny-mango.cfg', './darknet/data/yolov4-tiny-mango_final.weights',
                         '-ext_output', '-dont_show', '-out','./darknet/result.json'], stdin= open('./filenames.txt'))
if __name__ == '__main__':
    run_yolo();