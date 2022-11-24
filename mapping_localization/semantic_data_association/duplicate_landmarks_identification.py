import cv2
import os
import time
import pickle
import numpy as np
import sys
sys.path.append('/home/sam/git/Tracking_my_implementation')


start = time.time()
print("timer started \n")
import pickle
import cv2
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def duplicate_landmarks_identification(data_dir_pre, rerun_dis_stat = False, rerun_thresholding = True, visualize = False, average_div_by_threshold = 100):
        # TODO: 3D landmark based tracking dictionary key: tuple (3D landmark position), value is another dictionary as: {'track':the last track of this landmark, 'frame number': 1 + the last frame that observes this landmark}
    with open(data_dir_pre+'/landmark_pos_tracks_dict.pkl', 'rb') as input:
        landmark_pos_tracks_dict = pickle.load(input)

    if os.path.isfile(data_dir_pre+'/landmark_pairwise_distance.npy') == False or rerun_dis_stat:

        track_last_observe_fr_num_dict = {}
        landmark_diff_history = []
        landmark_distance = []
        total_iter = 0
        total_iter_expected = len(landmark_pos_tracks_dict) * (len(landmark_pos_tracks_dict) -1)
        for landmark_1 in landmark_pos_tracks_dict:
            print('cur_iter:',total_iter, 'expected_total_iter',total_iter_expected)
            for landmark_2 in landmark_pos_tracks_dict:
                total_iter += 1
                if landmark_2 != landmark_1:
                    landmark_diff = abs(np.asarray(landmark_2) - np.asarray(landmark_1))
                    # if landmark_diff not in landmark_diff_history:
                    landmark_distance.append(np.sqrt(landmark_diff[0] ** 2 + landmark_diff[1] ** 2 + landmark_diff[2] ** 2))
                    # landmark_diff_history.append(landmark_diff)
        landmark_distance_sorted = sorted(landmark_distance)
        np.save(data_dir_pre+'/landmark_pairwise_distance.npy', landmark_distance_sorted)

    duplicate_landmarks = {}
    landmark_distance_sorted = np.load(data_dir_pre+'/landmark_pairwise_distance.npy')
    length = len(landmark_distance_sorted)
    average = np.average(np.asarray(landmark_distance_sorted))
    distance_threshold = average / float(average_div_by_threshold)
    np.save(data_dir_pre+'/distance_threshold.npy',distance_threshold)

    already_read_landmarks = []
    already_added_dup = []
    total_iter = 0
    total_iter_expected = len(landmark_pos_tracks_dict) * (len(landmark_pos_tracks_dict) -1)


    num_dup_landmarks = 0
    if os.path.isfile(data_dir_pre+'/duplicate_landmarks_dict.pkl') == False or rerun_thresholding:
        for landmark_1, track_fr_num1 in landmark_pos_tracks_dict.items():
            if landmark_1 in already_added_dup:
                continue
            already_read_landmarks.append(landmark_1)
            last_observe_fr_num1 = track_fr_num1['frame number'] - 1
            print('cur_iter:',total_iter, 'expected_total_iter',total_iter_expected / 2)
            for landmark_2, track_fr_num2 in landmark_pos_tracks_dict.items():
                if landmark_2 not in already_read_landmarks and landmark_2 not in already_added_dup:
                    last_observe_fr_num2 = track_fr_num2['frame number'] - 1
                    total_iter += 1
                    landmark_diff = abs(np.asarray(landmark_2) - np.asarray(landmark_1))
                    # if landmark_diff not in landmark_diff_history:
                    landmark_12_distance = np.sqrt(landmark_diff[0] ** 2 + landmark_diff[1] ** 2 + landmark_diff[2] ** 2)
                    if landmark_12_distance < distance_threshold:
                        num_dup_landmarks += 1
                        # already_added_dup.append(landmark_1)
                        # already_added_dup.append(landmark_2)
                        print ('duplicate landmark found! Their diff is:', landmark_diff, 'num_dup_landmarks:',num_dup_landmarks)

                        if last_observe_fr_num1 > last_observe_fr_num2:
                            # TODO: in the form of: the duplicate (to remove) landmark : corresponding original (to keep) landmark
                            duplicate_landmarks[landmark_2] = landmark_1
                        else:
                            duplicate_landmarks[landmark_1] = landmark_2

        with open(data_dir_pre+'/duplicate_landmarks_dict.pkl', 'wb') as output:
            pickle.dump(duplicate_landmarks, output)

    else:
        print(data_dir_pre+'/duplicate_landmarks_dict.pkl found: duplicate landmarks are already recorded!')

    landmark_distance_sorted_small = landmark_distance_sorted[:int(length/5)]
    if visualize:
        count2 = plt.hist(landmark_distance_sorted_small, bins = 500,color = 'green')#bins=projected_area.shape[0]
        plt.plot([average/5.0,average / 5.0],[0,500],'blue',label = '1/5 of Mean Distance')
        plt.plot([average/float(average_div_by_threshold),average/ float(average_div_by_threshold)],[0,500],'red',label = 'Duplicate Threshold')
        plt.xlabel('Fruit Distance (Smallest 1/5)')
        plt.ylabel('Number of Fruit Pairs')
        plt.title('Fruit Distance') # (0-2000)
        plt.legend()
        plt.show(count2)




