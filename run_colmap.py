import subprocess
import os

def run_colmap(parent_data_dir, folder_dir_working_on):
    db_path = os.path.join(parent_data_dir,'generated_docs_this_run','database.db')
    image_path = os.path.join(parent_data_dir,'input')
    import_path = os.path.join(parent_data_dir, 'frame_features')
    matches_path = os.path.join(parent_data_dir, 'frame_match_idx', 'inp_match_idx.txt')
    bin_path =  os.path.join(parent_data_dir,'generated_docs_this_run', '0')
    output_path = os.path.join(parent_data_dir,'generated_docs_this_run')
    print("Creating DB:")
    if not os.path.exists(db_path):
        subprocess.call(['colmap', 'database_creator', '--database_path', db_path])
    print("Importing Features:")
    subprocess.call(['colmap', 'feature_importer', '--database_path', db_path, '--image_path', image_path,
                     '--import_path', import_path])
    subprocess.call(['colmap', 'matches_importer', '--database_path', db_path, '--match_list_path', matches_path,
                     '--match_type', 'raw'])
    subprocess.call(['colmap', 'mapper', '--database_path', db_path, '--image_path', image_path,
                     '--output_path', output_path])
    subprocess.call(['colmap', 'model_converter', '--input_path', bin_path, '--output_path', output_path,
                     '--output_type', 'TXT'])

if __name__ == '__main__':
    run_colmap(r'./test_data/', "")
