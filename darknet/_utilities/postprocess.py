import json
from PIL import Image, ImageDraw, ImageFont
from os.path import join, basename, splitext, isfile
from os import listdir, remove
import csv


# font = ImageFont.truetype("/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf", 11)
img_output_folder = '/home/tomvdon/darknet/build/darknet/x64/data/sample/predictions'
csv_output_folder = './test_data/det_all'
results_path = './darknet/result.json'

def clear():
    folders = [csv_output_folder]
    for folder in folders:
        files = listdir(folder)
        for f in files:
            if isfile(join(folder, f)):
                remove(join(folder,f))


def visualize():
    temp = open(results_path,)
    results = json.load(temp)
    for r in results:
        path = r['filename']
        im = Image.open(path)
        img_width, img_height = im.size
        bboxes = r['objects']
        for box in bboxes:
            x_center = box['relative_coordinates']['center_x'] * img_width
            y_center = box['relative_coordinates']['center_y'] * img_height
            width = box['relative_coordinates']['width'] * img_width
            height = box['relative_coordinates']['height'] * img_height
            conf = box['confidence']

            x_min = x_center - (width/2)
            x_max = x_min + width
            y_min = y_center - (height/2)
            y_max = y_min + height

            coords = [(x_min, y_min), (x_max, y_max)]
            d = ImageDraw.Draw(im)
            d.rectangle(coords, outline ="red")
            d.text((x_min, y_min - 12), str(conf)[:5], font = font, align ="left")
        print(join(img_output_folder, basename(path)))
        im.save(join(img_output_folder, basename(path)))

def convert_to_csvs():
    temp = open(results_path,)
    results = json.load(temp)
    for r in results:
        path = r['filename']
        filename, ext = splitext(basename(path))
        im = Image.open('./darknet/' + path)
        img_width, img_height = im.size
        bboxes = r['objects']
        with open(join(csv_output_folder, filename + '.csv'), 'w+') as f:
            w = csv.writer(f)
            for idx, box in enumerate(bboxes):
                x_center = box['relative_coordinates']['center_x'] * img_width
                y_center = box['relative_coordinates']['center_y'] * img_height
                width = box['relative_coordinates']['width'] * img_width
                height = box['relative_coordinates']['height'] * img_height
                conf = box['confidence']

                x_min = x_center - (width/2)
                x_max = x_min + width
                y_min = y_center - (height/2)
                y_max = y_min + height
                w.writerow([idx + 1, x_min, y_min, x_max, y_max, conf])
            f.close()

convert_to_csvs()
# clear()
# visualize()

