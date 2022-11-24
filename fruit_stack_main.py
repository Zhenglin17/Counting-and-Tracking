from tracking_stack_main import *
from run_yolo import *
from darknet._utilities.postprocess import *
import glob
import sys
import shutil

def get_file_names(source_folder, destination_folder):
    sorted_name_list = glob.glob(os.path.join(source_folder, '*.png'))
    textfile = open("filenames.txt", "w")
    for img_name in sorted_name_list:
        textfile.write(os.path.abspath(img_name) + "\n")
    textfile.close()
    ##Put in png
    directory = os.getcwd()
    path = os.path.join(directory,  destination_folder)
    files=os.listdir(source_folder)
    folder_path = os.path.join(path, "png")
    for fname in files:
        shutil.copy2(os.path.join(source_folder,fname), folder_path)

if __name__ == '__main__':
    source_folder = sys.argv[1]
    destination_folder = sys.argv[2]
    get_file_names(source_folder, destination_folder)
    run_yolo()
    convert_to_csvs(destination_folder)
    parent_data_dir = destination_folder
    folder_dir_working_on = destination_folder
    main(parent_data_dir, folder_dir_working_on)