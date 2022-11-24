Delete all data files in current folder
Run run.py in mapping_localization folder to generate initial tracking history
Copy the tracks_frame_idx_whole_list.pkl in mapping_localization folder to current folder, and run build_semantic_feature_matches.py
Use COLMAP to read the semantic feature matches (in frame_features and frame_match_idx folders) and generate initial recontracution
Export reconstruction as text files and copy them to mapping_localiztion/generated_docs_this_run
Run run_landmark_based_tracking_save_docs_for_SfM.py



