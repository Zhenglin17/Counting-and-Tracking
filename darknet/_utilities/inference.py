# ./darknet detector test build/darknet/x64/data/obj.data cfg/yolov4-tiny-mango.cfg build/darknet/x64/backup/yolov4-tiny-mango_final.weights /home/belinda/fruit/mmdetection/data/mangoes/sample-images/20171205T235240.045986.png -thres 0.3
# ./darknet detector test build/darknet/x64/data/obj.data cfg/yolov4-tiny-mango.cfg build/darknet/x64/backup/yolov4-tiny-mango_final.weights -ext_output -dont_show -thresh 0.8 -out result.txt < /home/belinda/yolo/darknet/build/darknet/x64/data/test.txt
import glob
import os
from os.path import isfile

cfg = 'build/darknet/x64/data/obj.data cfg/yolov4-tiny-mango.cfg'
weights = 'build/darknet/x64/backup/yolov4-tiny-mango_final.weights'
thres = .7

# input_dir = '/home/belinda/yolo/darknet/build/darknet/x64/data/sample'
input_dir = '/home/belinda/fruit/data/mangoes/sample-images'
file_ext = '.png'
output_path = '/home/belinda/yolo/darknet/output.txt'

with open(output_path, 'w') as output:
    for f in glob.iglob(os.path.join(input_dir, '*' + file_ext)):
        if isfile(os.path.join(input_dir, f)):
            title, ext = os.path.splitext(os.path.basename(f))
            output.write(input_dir + "/" + title + file_ext + "\n")

