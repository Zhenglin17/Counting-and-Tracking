# Fruit Tracking
## Dependencies and Installation Instructions 
Tested on Python 3.8.11, Ubuntu 20.04

1. Download and install colmap following instructions provided: https://colmap.github.io/  
2. Clone this fruit tracking repository.  
3. Clone and make darknet under the fruit stack repository following instructions provided:https://github.com/AlexeyAB/darknet TODO: Make this a subrepo when making public  
4. Install python requirements in a conda environment:   
```
 conda create --name fruit_stack --file requirements_conda.txt
 conda activate fruit_stack
 pip install -r requirements_pip.txt
```
## Models and Dataset
YOLOv4 Model can be found at: https://drive.google.com/drive/u/1/folders/1mt4fek0qEquT-KiUwzpMOVyd5A7BT8-I  
Save CFG file under darknet/cfg and weights under darknet/data  
Mango dataset can found at https://data.acfr.usyd.edu.au/ag/treecrops/2016-multifruit/   
Additional models can be trained using darknet instructions but requires changing filename in run_yolo.py  

## Instructions 
1. Initialize folder structure with 
```
python init.py <destination_folder>
```
Where destination_folder is the name of the root folder you wish to create  
2. Run full tracking stack with:  
```
python fruit_stack_main.py <source_folder> <destination_folder>
```
Where source_folder is a folder containing .PNG images you wish to track and destination_folder is the name of a folder created with init.py      
3. SFM model can be viewed using  
```
colmap gui
```
And importing model folder found under destination_folder/generated_docs_this_run

## Other Instructions

Note: 
(1) This code was written in a rush so it’s quite messy. We need to refactor it at some point. Just email me if you have any questions.
(2) The original tracking code written by Chao Qu has been open sourced (https://github.com/versatran01/scpye). Although it does not have interface with CNN predictions and does not provide files needed by SfM reconstruction, it can be a good reference for understanding. 
(3) Darknet branch with Utility functions to be added for creating CSVs.

Create your own data folder, place the data there(undistorted images in the png sub-folder and FR-CNN detections in ../det_all sub-folder);

Copy empty folders in copy_to_every_new_folder into the data folder you created;

Run acfr_draw_box_resize_pred_split_name_convert_to_number_idx.py;

Run run_2D_tracking.py (change data_dir_pre to your data folder directory) in generate initial tracking history;

Run build_semantic_feature_matches.py (change data_dir_pre to your data folder directory, and make sure the landmark_or_2Dtracking is set to '2D');

Use COLMAP to read the semantic feature matches (in frame_features and frame_match_idx folders in your data folder directory) as RAW feature matches and generate initial recontracution;

Export reconstruction as text files and copy them to your_data_folder_directory/generated_docs_this_run;

Run run_landmark_based_tracking_save_docs_for_SfM.py;


Detailed:
localize trunk corners section:
***TODO: if segmentation results are out, copy the segmentation result images (one per tree) to trunk_img_seg/trunk_segmentation folder (delete other files in trunk_img_seg), copy corresponding 5 images to that folder

1. find the corresponding timestamp folder in ACFR_RCNN_GT*/todo folder, copy to ACFR_RCNN_GT*/doing
2. choose five good images in the 'input' folder (in the above dir) to trunk_img_seg
3. run temp_contrast_enhance.py
4. run ACFR_trunk_label.m 
5. run run_landmark_tracking.py


Updated code after RA-L:
Landmark_fruit_tracker.py: landmark_horizon [-20,-1] to [-30,-1]


------------
