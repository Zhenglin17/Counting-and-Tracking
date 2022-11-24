import os
import sys
import shutil
def create_folders(foldername):
    directory = os.getcwd()
    path =  os.path.join(directory, foldername)
    folders = ["check_mask_correctness_for_landmark_tracking",
    "det_all",
 "frame_features",
 "frame_match_idx",
 "generated_docs_this_run",
 "input",
 "landmark_counting_results",
"landmark_counting_visulization",
"landmark_tracking_results",
"landmark_tracking_results_idx_sync",
"png",
"pred",
"pred_bbox",
"projection_results",
"semantic_features_results",
"tracking_results",
"tracking_results_masked",
"tracking_results_masked_visulization"]
    for folder in folders:
        folder_path = os.path.join(path, folder)
        os.makedirs(folder_path)
        print(f"Created {folder_path}")
if __name__ == "__main__":
    create_folders(sys.argv[1])



