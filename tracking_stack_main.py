from acfr_draw_box_resize_pred_split_name_convert_to_number_idx import acfr_draw_boxes
from run_2D_tracking import two_dim_tracking
from build_semantic_feature_matches import semantic_feature_matches
from run_colmap import run_colmap
from run_landmark_tracking import run_landmark_tracking
def main(parent_data_dir, folder_dir_working_on):
    acfr_draw_boxes(parent_data_dir, folder_dir_working_on)
    two_dim_tracking(parent_data_dir, folder_dir_working_on)
    semantic_feature_matches(parent_data_dir, folder_dir_working_on)
    run_colmap(parent_data_dir, folder_dir_working_on)
    # run_landmark_tracking(parent_data_dir, folder_dir_working_on)




if __name__ == '__main__':
    parent_data_dir = r'./test_data/'
    folder_dir_working_on = r'./test_data/'
    main(parent_data_dir, folder_dir_working_on)