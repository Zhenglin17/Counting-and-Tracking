import glob, os
from os import listdir, remove
from os.path import join, isfile
from posixpath import splitext
import shutil
import csv

# input_dir = os.path.dirname(os.path.abspath(__file__))
input_dir = '/home/belinda/fruit/data/mangoes/images'
output_dir = 'data/obj' # path relative to darknet.exe
abs_output_dir = '/home/belinda/yolo/darknet/build/darknet/x64/data'
file_ext = '.png'
percentage_test = 15; # test set percentage rel. to full set

# clean the folders
# def clean():
#     folders = [output_dir, join(output_dir, 'obj')]
#     for folder in folders:
#         files = listdir(folder)
#         for f in files:
#             if isfile(join(folder, f))
#                 remove(join(folder,f))

def create_obj_files():
    obj_names = 'mango'
    obj_data = ('classes = 1\n'
               'train = /home/belinda/yolo/darknet/build/darknet/x64/data/train.txt\n'
               'valid = /home/belinda/yolo/darknet/build/darknet/x64/data/test.txt\n'
               'names = /home/belinda/yolo/darknet/build/darknet/x64/data/obj.names\n'
               'backup = /home/belinda/yolo/darknet/build/darknet/x64/backup')
    temp = [(obj_names, 'obj.names'), (obj_data, 'obj.data')]
    for data, filename in temp:
        with open(join(abs_output_dir, filename), 'w') as f:
            f.write(data)

def move_images_annotations():
    print('copying ' + str(len(listdir(input_dir))) + ' images to ' + abs_output_dir)
    img_width = 500
    img_height = 500
    # move pngs
    for img in listdir(input_dir):
        shutil.copy(join(input_dir, img), join(abs_output_dir + '/obj', img))
    # annotations are in 3 folders, unfortunately
    for anno_folder in ['/home/belinda/fruit/data/mangoes/test', 
                        '/home/belinda/fruit/data/mangoes/train',
                        '/home/belinda/fruit/data/mangoes/val']:
        for anno in listdir(anno_folder):
            with open(join(anno_folder, anno), newline='') as f:
                data = list(csv.reader(f))
                output = join(abs_output_dir + '/obj', splitext(anno)[0] + '.txt')
                with open(output, 'w') as g:
                    if len(data) > 1:
                        for bbox in data[1:]:

                            x = float(bbox[1])
                            y = float(bbox[2])
                            dx = float(bbox[3])
                            dy = float(bbox[4])

                            x_center = (x + dx/2) / img_width
                            y_center = (y + dy/2) / img_height
                            height = dy / img_height
                            width = dx / img_height
                            yolo_data = ('0 ' + str(x_center) + 
                                        ' ' + str(y_center) + 
                                        ' ' + str(height) + 
                                        ' ' + str(width) + '\n')
                            g.write(yolo_data)
                g.close()
                f.close()

def create_train_test():
    file_train = open('/home/belinda/yolo/darknet/build/darknet/x64/data/train.txt', 'w')
    file_test = open('/home/belinda/yolo/darknet/build/darknet/x64/data/test.txt', 'w')

    # populate train.txt and test.txt
    counter = 1
    index_test = round(100 / percentage_test)
    for f in glob.iglob(os.path.join(input_dir, '*' + file_ext)):
        title, ext = os.path.splitext(os.path.basename(f))

        if counter == index_test:
            counter = 1
            file_test.write(abs_output_dir + "/obj/" + title + file_ext + "\n")
        else:
            file_train.write(abs_output_dir + "/obj/" + title + file_ext + "\n")
            counter = counter + 1
    
    file_train.close()
    file_test.close()


create_obj_files()
move_images_annotations()
create_train_test()

# train
# ./darknet detector train build/darknet/x64/data/obj.data cfg/yolov4-tiny-mango.cfg build/darknet/x64/yolov4-tiny.conv.29 

# test
# ./darknet detector test build/darknet/x64/data/obj.data cfg/yolov4-tiny-mango.cfg build/darknet/x64/backup/yolov4-tiny-mango_final.weights /home/belinda/fruit/mmdetection/data/mangoes/sample-images/20171205T235240.045986.png -thres 0.3