from __future__ import (print_function, division, absolute_import)
import pickle
import cv2
import numpy as np
import sys
import transforms3d as trans3d
import os
from scpye.improc.image_processing import enhance_contrast
from scpye.track.assignment import hungarian_assignment
from scpye.track.bounding_box import (bboxes_assignment_cost, extract_bbox)
from scpye.track.fruit_track import FruitTrack
from scpye.track.optical_flow import (calc_optical_flow, calc_average_flow)
from scpye.utils.drawing import (Colors, draw_bboxes, draw_optical_flows,
                                 draw_text, draw_bboxes_matches, draw_line, draw_bbox_centers)
from scpye.track.bounding_box import (bbox_center, shift_bbox)
from mapping_localization.semantic_data_association.extract_landmark_3D_pos_bboxes_hist import  extract_landmark_3D_pos_bboxes_hist
from mapping_localization.semantic_data_association.extract_every_frame_rot_trans import  extract_every_frame_rot_trans
import copy
from scpye.utils.drawing import (Colors, draw_bboxes, draw_optical_flows,
                                 draw_text, draw_text_mask, draw_bboxes_matches, draw_line, draw_bbox_centers, draw_bbox_as_circles)

import time

def project_landmarks_add_unseen_observation(current_frame_idx, landmark_horizon, img_w_h, fruit_3D_pos_bboxes_hist_dict, rot_trans_dict, K_mat, data_dir_pre):
    bad_case = 0
    start = time.time()
    good_3D_point_idx = []
    if str(current_frame_idx) not in rot_trans_dict:
        raise Exception('camera pose for frame', current_frame_idx,'is not in rot_trans_dict (not reconstructed)!\n' )
    rot = rot_trans_dict[str(current_frame_idx)][0]
    trans = rot_trans_dict[str(current_frame_idx)][1]
    projected_fruits_bboxes_with_hist_3D_pos_dict = {}
    bboxes_proj = []
    for three_dim_pt_idx, value in fruit_3D_pos_bboxes_hist_dict.items():
        # fruit_3D_pos_bboxes_hist_dict: key is 3D point index, value is a dictionary: {'3Dpos':fruit_3D_loc, 'frame_idx': [frame_ind] (order: small to large), '2Dpos_bbox':fruit}, which contains fruits' 3D position, frame index where it appears (a list of integers) and corresponding frame's 2D_position_and_bbox (a list where each element is another list: [array(x_center, y_center),array(bbox)])
        #fruit_3D_pos_bboxes_hist_dict[three_dim_pt_idx_all[0]] = {'3Dpos':fruit_3D_loc, 'frame_idx': frame_ind, '2Dpos_bbox':fruit}
        # Note: fruit 2D position is based on the Kalman Filter output, not the FRCNN bbox center!!!
        # only look at landmarks (fruits) that has already been initialized to 3D from frames before the current frame
        if value['frame_idx'][0] <= current_frame_idx + landmark_horizon[1] and value['frame_idx'][0]  >= current_frame_idx + landmark_horizon[0]:
            # points locations in the global coordinate
            pos_world_frame = value['3Dpos']
            # points locations in the camera coordinate
            pos_cam_frame = np.dot(rot, pos_world_frame) + trans
            # points locations in the image coordinate
            pos_img = np.dot(K_mat, pos_cam_frame)
            # normalize
            pos_img /= pos_img[2]
            x_img = pos_img[0]
            y_img = pos_img[1]
            new_center = [x_img, y_img]
            # extract the latest bbox and shift it to current center
            # fruit_3D_pos_bboxes_hist_dict['2Dpos_bbox'][idx]: list of two arrays, first is (2,) fruit center, second is (4,) bounding box. Note: fruit 2D position is based on the Kalman Filter output, not the FRCNN bbox center!!!


            cloest_bbox = value['2Dpos_bbox'][-1][1]
            # check if the bounding box projection is out of the image boundary
            if x_img > img_w_h[0] - 1 - cloest_bbox[2] / 2.0 or x_img < cloest_bbox[2] / 2.0 or y_img > img_w_h[1] - 1 - cloest_bbox[3] / 2.0 or y_img < cloest_bbox[3] / 2.0:
                # print('current bounding box projection is out of current frame image boundary, its center position is', new_center)
                continue

            else:
                # if there are some frames which are before current frame but have no history of the landmark, it means the detector fails to detect it, hereby I add the estimation of them back to the image
                current_unobserved_bboxes_fridx_dict = {}
                if value['frame_idx'][-1] < current_frame_idx-1:
                    # print('unseen frames found for landmark:',three_dim_pt_idx,', the most recent detection was in frame:', value['frame_idx'][-1], 'but current frame is:', current_frame_idx,'. Hereby add the predicted observations back!')
                    for unseen_fr_idx in np.arange(value['frame_idx'][-1]+1, current_frame_idx):
                            if str(unseen_fr_idx) not in rot_trans_dict:
                                print('check rot_trans_dict, the frame',unseen_fr_idx,' is probably not in there (thus not reconstructed)')
                            rot_unseen = rot_trans_dict[str(unseen_fr_idx)][0]
                            trans_unseen = rot_trans_dict[str(unseen_fr_idx)][1]
                            # points locations in the camera coordinate
                            pos_cam_frame_unseen = np.dot(rot_unseen, pos_world_frame) + trans_unseen
                            # points locations in the image coordinate
                            pos_img_unseen = np.dot(K_mat, pos_cam_frame_unseen)
                            # normalize
                            pos_img_unseen /= pos_img_unseen [ 2]
                            x_img_unseen = pos_img_unseen[0]
                            y_img_unseen = pos_img_unseen[1]
                            new_center_unseen = [x_img_unseen, y_img_unseen]
                            # check if the bounding box projection is out of the image boundary
                            if x_img_unseen > img_w_h[0] - 1 - cloest_bbox[2] / 2.0 or x_img_unseen < 0 or y_img_unseen > img_w_h[1] - 1 - cloest_bbox[3] / 2.0 or y_img_unseen < 0:
                                print('\nthis bounding box projection is out of this USEEN frame image boundary, its center position is', new_center_unseen, '. This is abnormal since the bounding box projection, whose center is:',new_center,', is inside the boundary of the current frame (a subsequent frame for this useen frame)! This useen frame index and corresponding current frame index are:', unseen_fr_idx, current_frame_idx,'\n')
                                if x_img_unseen > img_w_h[0] - 1 - cloest_bbox[2] / 2.0:
                                    x_img_unseen = img_w_h[0] - 1 - cloest_bbox[2] / 2.0
                                elif x_img_unseen < 0:
                                    x_img_unseen = 0
                                elif y_img_unseen > img_w_h[1] - 1 - cloest_bbox[3] / 2.0:
                                    y_img_unseen = img_w_h[1] - 1 - cloest_bbox[3] / 2.0
                                elif y_img_unseen < 0:
                                    y_img_unseen = 0
                                new_center_unseen = [x_img_unseen, y_img_unseen]
                                unseen_bbox_shifted_to_proj_center = shift_bbox(cloest_bbox, new_center_unseen)
                                current_unobserved_bboxes_fridx_dict[unseen_fr_idx] = [np.array(new_center_unseen), unseen_bbox_shifted_to_proj_center]
                            else:
                                unseen_bbox_shifted_to_proj_center = shift_bbox(cloest_bbox, new_center_unseen)
                                current_unobserved_bboxes_fridx_dict[unseen_fr_idx] = [np.array(new_center_unseen), unseen_bbox_shifted_to_proj_center]

                # record this current fruit in current frame, and PREDICT its position in the middle frames which it should have appeared but just was not detected
                good_3D_point_idx.append(three_dim_pt_idx)
                bbox_shifted_to_proj_center = shift_bbox(cloest_bbox, new_center)
                # projected_fruits_bboxes_with_hist_3D_pos_dict: projected_fruits_bboxes_with_hist_3D_pos_dict[three_dim_pt_idx] = {'3Dpos':value['3Dpos'], 'proj_bbox':bbox_shifted_to_proj_center, 'frame_idx':value['frame_idx'], '2Dpos_bbox':value['2Dpos_bbox'], 'unobserved':[True means the fruit bbox is not observed but predicted by projecting the landmark back]}, where value is from fruit_3D_pos_bboxes_hist_dict,
                # specifically  '3Dpos':fruit_3D_loc, 'frame_idx': [frame_ind] (order: small to large), '2Dpos_bbox':fruit}, which contains fruits' 3D position, frame index where it appears (a list of integers) and corresponding frame's 2D_position_and_bbox (a list where each element is another list: [array(x_center, y_center),array(bbox)])
                #Note: fruit 2D position is based on the Kalman Filter output, not the FRCNN bbox center!!!
                unobserved = [False for element in value['2Dpos_bbox']]
                projected_fruits_bboxes_with_hist_3D_pos_dict[three_dim_pt_idx] = {'3Dpos':copy.deepcopy(value['3Dpos']), 'proj_bbox':copy.deepcopy(bbox_shifted_to_proj_center), 'frame_idx':copy.deepcopy(value['frame_idx']), '2Dpos_bbox':copy.deepcopy(value['2Dpos_bbox']), 'unobserved':copy.deepcopy(unobserved)}
                bboxes_proj.append(bbox_shifted_to_proj_center)
                for unseen_fr_idx in np.arange(value['frame_idx'][-1]+1, current_frame_idx):
                    if unseen_fr_idx in current_unobserved_bboxes_fridx_dict:
                        if unseen_fr_idx != projected_fruits_bboxes_with_hist_3D_pos_dict[three_dim_pt_idx]['frame_idx'][-1] + 1:
                            raise Exception('frame index not consistent while adding unseen observations, check the implementation!')
                        projected_fruits_bboxes_with_hist_3D_pos_dict[three_dim_pt_idx]['frame_idx'].append(copy.deepcopy(unseen_fr_idx))
                        projected_fruits_bboxes_with_hist_3D_pos_dict[three_dim_pt_idx]['2Dpos_bbox'].append(copy.deepcopy(current_unobserved_bboxes_fridx_dict[unseen_fr_idx]))
                        projected_fruits_bboxes_with_hist_3D_pos_dict[three_dim_pt_idx]['unobserved'].append(True)
                        # print('current unseen fr idx', unseen_fr_idx)


                old_fr_idx = []
                bad_idxes = []
                # check whether there are duplicated projections of the same landmark in the same frame
                for idx, fr_idx in enumerate(projected_fruits_bboxes_with_hist_3D_pos_dict[three_dim_pt_idx]['frame_idx']):
                    if fr_idx in old_fr_idx:
                        bad_idxes.append(idx)
                        print('there are duplicated projections of the same landmark in the same frame, all fr_idx in the list are:',projected_fruits_bboxes_with_hist_3D_pos_dict[three_dim_pt_idx]['frame_idx'])
                    old_fr_idx.append(fr_idx)
                for bad_idx in bad_idxes:
                    del projected_fruits_bboxes_with_hist_3D_pos_dict[three_dim_pt_idx]['frame_idx'][bad_idx]
                    del projected_fruits_bboxes_with_hist_3D_pos_dict[three_dim_pt_idx]['2Dpos_bbox'][bad_idx]
                    del projected_fruits_bboxes_with_hist_3D_pos_dict[three_dim_pt_idx]['unobserved'][bad_idx]

                    print('after duplicates removal, all fr_idx in the list are:',projected_fruits_bboxes_with_hist_3D_pos_dict[three_dim_pt_idx]['frame_idx'])

                # check whether there are missing features in the middle frames (dropped out by sfm becasue of large reproj error), if yes, add them back
                start_fr_idx = projected_fruits_bboxes_with_hist_3D_pos_dict[three_dim_pt_idx]['frame_idx'][0]
                end_fr_idx = projected_fruits_bboxes_with_hist_3D_pos_dict[three_dim_pt_idx]['frame_idx'][-1]
                estimated_length = len(projected_fruits_bboxes_with_hist_3D_pos_dict[three_dim_pt_idx]['frame_idx'])
                if start_fr_idx+estimated_length-1 != end_fr_idx:
                    actual_length = end_fr_idx - start_fr_idx + 1

                    # if actual_length - estimated_length > 8:
                    #     print('\n\nToo many unobserved frames, hereby delete this 3D point, its index is ', three_dim_pt_idx,'!\n\n')
                    #     del projected_fruits_bboxes_with_hist_3D_pos_dict[three_dim_pt_idx]
                    #     continue

                    # print('total unobserved cases to add for this 3D landmark:',actual_length - estimated_length)
                    if (actual_length - estimated_length) < 1:
                        raise Exception('total unobserved cases to add for this 3D landmark is less than 1, check implementation!')
                    prev_fr_idx = start_fr_idx
                    pos_world_frame = projected_fruits_bboxes_with_hist_3D_pos_dict[three_dim_pt_idx]['3Dpos']
                    for idx in np.arange(1,actual_length):
                        if projected_fruits_bboxes_with_hist_3D_pos_dict[three_dim_pt_idx]['frame_idx'][idx] != prev_fr_idx + 1:
                            unseen_fr_idx = prev_fr_idx + 1
                            if str(unseen_fr_idx) not in rot_trans_dict:
                                print('check rot_trans_dict, the frame',unseen_fr_idx,' is probably not in there (thus not reconstructed)')
                            rot_unseen = rot_trans_dict[str(unseen_fr_idx)][0]
                            trans_unseen = rot_trans_dict[str(unseen_fr_idx)][1]
                            # points locations in the camera coordinate
                            pos_cam_frame_unseen = np.dot(rot_unseen, pos_world_frame) + trans_unseen
                            # points locations in the image coordinate
                            pos_img_unseen = np.dot(K_mat, pos_cam_frame_unseen)
                            # normalize
                            pos_img_unseen /= pos_img_unseen [ 2]
                            x_img_unseen = pos_img_unseen[0]
                            y_img_unseen = pos_img_unseen[1]
                            new_center_unseen = [x_img_unseen, y_img_unseen]
                            # check if the bounding box projection is out of the image boundary
                            cloest_bbox = projected_fruits_bboxes_with_hist_3D_pos_dict[three_dim_pt_idx]['2Dpos_bbox'][idx-1][1]
                            if x_img_unseen > img_w_h[0] - 1 - cloest_bbox[2] / 2.0 or x_img_unseen < 0 or y_img_unseen > img_w_h[1] - 1 - cloest_bbox[3] / 2.0 or y_img_unseen < 0:
                                print('this bounding box projection is out of this USEEN frame image boundary, its center position is', new_center_unseen, '. This is abnormal since the bounding box projection, whose center is:',new_center,', is inside the boundary of the current frame (a subsequent frame for this useen frame)! This useen frame index and corresponding current frame index are:', unseen_fr_idx, current_frame_idx)
                                if x_img_unseen > img_w_h[0] - 1 - cloest_bbox[2] / 2.0:
                                    x_img_unseen = img_w_h[0] - 1 - cloest_bbox[2] / 2.0
                                elif x_img_unseen < 0:
                                    x_img_unseen = 0
                                elif y_img_unseen > img_w_h[1] - 1 - cloest_bbox[3] / 2.0:
                                    y_img_unseen = img_w_h[1] - 1 - cloest_bbox[3] / 2.0
                                elif y_img_unseen < 0:
                                    y_img_unseen = 0
                                new_center_unseen = [x_img_unseen, y_img_unseen]
                                unseen_bbox_shifted_to_proj_center = shift_bbox(cloest_bbox, new_center_unseen)
                                projected_fruits_bboxes_with_hist_3D_pos_dict[three_dim_pt_idx]['frame_idx'].insert(idx, prev_fr_idx + 1)
                                # print('current inserted index:',prev_fr_idx + 1)
                                projected_fruits_bboxes_with_hist_3D_pos_dict[three_dim_pt_idx]['2Dpos_bbox'].insert(idx, copy.deepcopy([np.array(new_center_unseen), unseen_bbox_shifted_to_proj_center]))
                                projected_fruits_bboxes_with_hist_3D_pos_dict[three_dim_pt_idx]['unobserved'].insert(idx, True)

                            else:
                                unseen_bbox_shifted_to_proj_center = shift_bbox(cloest_bbox, new_center_unseen)
                                projected_fruits_bboxes_with_hist_3D_pos_dict[three_dim_pt_idx]['frame_idx'].insert(idx, prev_fr_idx + 1)
                                # print('current inserted index:',prev_fr_idx + 1)
                                projected_fruits_bboxes_with_hist_3D_pos_dict[three_dim_pt_idx]['2Dpos_bbox'].insert(idx, copy.deepcopy([np.array(new_center_unseen), unseen_bbox_shifted_to_proj_center]))
                                projected_fruits_bboxes_with_hist_3D_pos_dict[three_dim_pt_idx]['unobserved'].insert(idx, True)

                        if len(projected_fruits_bboxes_with_hist_3D_pos_dict[three_dim_pt_idx]['unobserved']) != len(projected_fruits_bboxes_with_hist_3D_pos_dict[three_dim_pt_idx]['2Dpos_bbox']) or len(projected_fruits_bboxes_with_hist_3D_pos_dict[three_dim_pt_idx]['2Dpos_bbox']) != len(projected_fruits_bboxes_with_hist_3D_pos_dict[three_dim_pt_idx]['frame_idx']):
                            raise Exception('check implamentation of project_landmarks.py')
                        prev_fr_idx = projected_fruits_bboxes_with_hist_3D_pos_dict[three_dim_pt_idx]['frame_idx'][idx]

                    # print('current 3D landmark:',three_dim_pt_idx,'\n')
                    flag = 0
                    check_prev_idx = start_fr_idx
                    for check_idx in np.arange(1,len(projected_fruits_bboxes_with_hist_3D_pos_dict[three_dim_pt_idx]['frame_idx'])):
                        if projected_fruits_bboxes_with_hist_3D_pos_dict[three_dim_pt_idx]['frame_idx'][check_idx] != check_prev_idx +1:
                            raise Exception('frame index is still not consecutive for current 3D landmark, check the implementation!')
                        check_prev_idx = projected_fruits_bboxes_with_hist_3D_pos_dict[three_dim_pt_idx]['frame_idx'][check_idx]
                    bad_case += 1
                    continue


    visualize_projection_results  = 1
    if visualize_projection_results:
        img_idx = current_frame_idx
        if os.path.isfile(data_dir_pre+'/pred_bbox/pred_frame'+('%04d' %img_idx)+'.png') == False:
             raise Exception(data_dir_pre+'/pred_bbox/pred_frame'+('%04d' %img_idx)+'.png does not exist!!')
        cur_img = cv2.imread(data_dir_pre+'/pred_bbox/pred_frame'+('%04d' %img_idx)+'.png')
        # for x_img, y_img, bbox in zip(xs_img,ys_img, bboxes_proj):

        if bboxes_proj != []:
            draw_bbox_as_circles(cur_img, bboxes_proj, color=[220,120,120], thickness=2)
            # draw_bboxes(cur_img, bboxes_proj, color=[220,120,120], thickness=2)
        h,w,_ = cur_img.shape
        draw_text = False
        if draw_text:
            draw_text_mask(cur_img, '', 'Green: fruits detected in current image.', (10, h - 20), scale=1.5,
                  color=Colors.cyan)
            draw_text_mask(cur_img, '', 'Purple: landmarks projected to current image', (10, h - 70), scale=1.5,
                  color=Colors.cyan)
        cv2.imwrite(data_dir_pre+'/projection_results/projection_visualization_for_frame'+('%04d' %img_idx)+'.png',cur_img)
        # cur_show = cv2.resize(cur_img,(0,0),fx=0.5, fy = 0.5)
        # cv2.imshow('projection results:',cur_show)
        # cv2.waitKey(100)
        # cv2.destroyAllWindows()




    end = time.time()
    print('running time for project_landmarks_add_unseen_observation of current frame is:',end-start)
    print('number of all 3D landmarks (of frames in the horizon) is:', len(fruit_3D_pos_bboxes_hist_dict), 'number of 3D landmarks projected to current frame is', len(good_3D_point_idx))
    print('number of cases that do not have a consecutive history:',bad_case)
    return projected_fruits_bboxes_with_hist_3D_pos_dict

    # # TODO: find the projections of all 3D landmarks in every image, save the bboxes_with_hist_dict, where the key is the frame index and value is a list where first element: bounding box info, second element: bounding box history info
    # with open('../generated_docs_this_run/image_and_camera_id.txt') as f:
    #     # Image list with two lines of data per image:
    #     #   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID
    #     #   POINTS2D[] as (X, Y, POINT3D_ID)
    #     bboxes_with_hist_dict = {}
    #     bboxes_proj = []
    #     lines=f.readlines()
    #     ind = -1
    #     length = len(lines) - 1
    #     img_pos_id = np.zeros((length,4))
    #     img_pos_id_list = []
    #     for line in lines:
    #         # skip comments in the first four lines
    #         if ind >= 0:
    #             # image pose information
    #             img_info_arr = (np.fromstring(line, dtype=float, sep=' '))
    #             quat_img = img_info_arr[1:5]
    #             rot_mat_world_to_img = trans3d.quaternions.quat2mat(quat_img)
    #             trans_world_to_img = img_info_arr[5:8]
    #             frame_id_temp = int(img_info_arr[0])
    #             rot_trans_dict[str(frame_id_temp)] = [rot_mat_world_to_img,trans_world_to_img]
    #         ind += 1
    #     num_1 = frame_ids.shape[0]
    #     pos_cam_frame = np.zeros((num_1,3))
    #     pos_img = np.zeros((num_1,3))
    #     # points locations in the global coordinate
    #     xs_cam_frame = np.zeros_like(xs)
    #     ys_cam_frame = np.zeros_like(ys)
    #     zs_cam_frame = np.zeros_like(zs)
    #
    #     # points locations in the image coordinate
    #     xs_img = np.zeros_like(xs)
    #     ys_img = np.zeros_like(ys)
    #     pos_cam_frame_area_fridx_of_fruits = fruits_pos_area_arr
    #     for pt_idx in range(num_1):
    #         frid_int = int(frame_ids[pt_idx])
    #         #TODO*****************************************************************************************************************************************************
    #         if frid_int != 50:
    #             frid_int = 50
    #         # TODO: here I directly let camera poses which are not obtained to be its neighbor (last frame) camera poses!!!!!!!!!!!!!!!!!! Maybe change this !!!!!!
    #         if str(frid_int) not in rot_trans_dict:
    #             print('point',pt_idx,'corresponding frame index:', frid_int, ' camera pose is not reconstructed, continue to the next frame... \n' )
    #         rot = rot_trans_dict[str(frid_int)][0]
    #         trans = rot_trans_dict[str(frid_int)][1]
    #         # points locations in the global coordinate
    #         pos_world_frame = np.array([xs[pt_idx],ys[pt_idx],zs[pt_idx]])
    #         # points locations in the camera coordinate
    #         pos_cam_frame[pt_idx,:] = np.dot(rot, pos_world_frame) + trans
    #         xs_cam_frame[pt_idx] = pos_cam_frame[pt_idx, 0]
    #         ys_cam_frame[pt_idx] = pos_cam_frame[pt_idx, 1]
    #         zs_cam_frame[pt_idx] = pos_cam_frame[pt_idx, 2]
    #         # points locations in the image coordinate
    #         pos_img[pt_idx,:] = np.dot(K_mat, pos_cam_frame[pt_idx, :])
    #         # normalize
    #         pos_img[pt_idx,:] /= pos_img [pt_idx, 2]
    #         xs_img[pt_idx] = pos_img[pt_idx,0]
    #         ys_img[pt_idx] = pos_img[pt_idx,1]
    #         new_center = [xs_img[pt_idx], ys_img[pt_idx]]
    #         # record corresponding bbox info, and move in current image plane
    #         prev_bbox = frame_bboxes[pt_idx]
    #         bbox_proj = shift_bbox(prev_bbox, new_center)
    #         bboxes_proj.append(bbox_proj)
    #         bboxes_with_hist = bboxes_proj, bboxes_hist
    #         bboxes_with_hist_dict[frid_int].append(bboxes_with_hist)
    #
    #         lost_fruits_bboxes_detect = [fruit[1] for fruit in fruits]
    #
    #         # cost = bboxes_assignment_cost(bboxes_proj, lost_fruits_bboxes_detect)
    #         # match_inds, lost_inds, new_inds = hungarian_assignment(cost, unassigned_cost=hungarian_unassign_cost)
    #
    #         if pos_img [pt_idx,2] != 1:
    #             raise Exception ('check normalization of image coordinate (3rd element should = 1)')
    #
    #         pos_cam_frame_area_fridx_of_fruits[pt_idx,0:3] = pos_cam_frame[pt_idx,:].T
    #         # area3D = area2D * depth^2
    #         pos_cam_frame_area_fridx_of_fruits[pt_idx, 3] = pos_cam_frame_area_fridx_of_fruits[pt_idx, 3] * (pos_cam_frame_area_fridx_of_fruits[pt_idx, 2])**2
    #
    #     np.save('../generated_docs_this_run/pos_cam_frame_area_fridx_of_fruits.npy',pos_cam_frame_area_fridx_of_fruits)
    #     np.savetxt('../generated_docs_this_run/pos_cam_frame_area_fridx_of_fruits.txt',pos_cam_frame_area_fridx_of_fruits,fmt = '%.2f',header='3D positions in camera frame and areas*depth^2(z^2) of fruits, format: X, Y, Z, bbox area*Z^2, fruit frame ID')
    #
    #     if visualize_projection_results:
    #         img_idx = 50
    #         if os.path.isfile('current image name: /home/sam/Desktop/Fruit-count-dataset/ACFR_RCNN_resized/input/frame'+('%04d' %img_idx)+'.png') == False:
    #              print('current image name: /home/sam/Desktop/Fruit-count-dataset/ACFR_RCNN_resized/input/frame'+('%04d' %img_idx)+'.png does not exist!!')
    #         cur_img = cv2.imread('./pred_frame0050.png')#'/home/sam/Desktop/Fruit-count-dataset/ACFR_RCNN_resized/input/frame'+('%04d' %img_idx)+'.png')
    #         # for x_img, y_img, bbox in zip(xs_img,ys_img, bboxes_proj):
    #             # cur_img[int(y_img)-2:int(y_img)+2, int(x_img)-2:int(x_img)+2,:] = [0, 0, 255]
    #         draw_bboxes(cur_img, bboxes_proj, color=Colors.cyan, thickness=1, margin=1)
    #         cv2.imwrite('./projection_visualization_for_50th_frame.png',cur_img)
    #         cur_show = cv2.resize(cur_img,(0,0),fx=0.75, fy = 0.75)
    #         cv2.imshow('projection results:',cur_show)
    #         cv2.waitKey(1000)
    #         cv2.destroyAllWindows()


if __name__ == "__main__":


    data_dir_pre = '/home/sam/Desktop/Fruit-count-dataset/ACFR_RCNN_GT/20171206T042534.518455'
    # fruit_3D_pos_bboxes_hist_dict: key is 3D point index, value is a dictionary: {'3Dpos':fruit_3D_loc, 'frame_idx': [frame_ind] (order: small to large), '2Dpos_bbox':fruit}, which contains fruits' 3D position, frame index where it appears (a list of integers) and corresponding frame's 2D_position_and_bbox (a list where each element is another list: [array(x_center, y_center),array(bbox)])
    fruit_3D_pos_bboxes_hist_dict_fname = data_dir_pre+'/generated_docs_this_run/fruit_3D_pos_bboxes_hist_dict.pkl'
    # every_frame_rot_trans_dict is recorded as follows: every_frame_rot_trans_dict[str(frame_id_temp)] = [rot_mat_world_to_img,trans_world_to_img]
    rot_trans_dict_fname = data_dir_pre+'/generated_docs_this_run/every_frame_rot_trans_dict.pkl'
    if os.path.isfile(fruit_3D_pos_bboxes_hist_dict_fname) == False:
        print(fruit_3D_pos_bboxes_hist_dict_fname + 'does not exist, running extract_landmark_3D_pos_bboxes_hist.py to generate the file\n')
        extract_landmark_3D_pos_bboxes_hist()
    else:
        print(fruit_3D_pos_bboxes_hist_dict_fname + 'already exists, NOT re-running extract_landmark_3D_pos_bboxes_hist.py\n')

    if os.path.isfile(rot_trans_dict_fname) == False:
        print(rot_trans_dict_fname + 'does not exist, running extract_every_frame_rot_trans.py to generate the file\n')
        extract_every_frame_rot_trans()
    else:
        print(rot_trans_dict_fname + 'already exists, NOT re-running extract_every_frame_rot_trans.py\n')

    with open(fruit_3D_pos_bboxes_hist_dict_fname, 'rb') as input:
        fruit_3D_pos_bboxes_hist_dict = pickle.load(input)
    with open(rot_trans_dict_fname, 'rb') as input:
        rot_trans_dict = pickle.load(input)

    with open(data_dir_pre+'/generated_docs_this_run/cameras.txt') as f:
        lines=f.readlines()
        for line in lines:
            # skip the comments
            if line[0] == '#':
                continue
            else:
                if '\n' in line:
                    line = line [:-1]
                cam_info = line.split(' ')#np.fromstring(line, dtype=float, sep='\n')
                if len(cam_info) != 8:
                    print(len(cam_info))
                    raise Exception('camera info not correct, check cameras.txt!')
    f_x =float(cam_info[4])
    f_y =float(cam_info[5])
    c_x =float(cam_info[6])
    c_y =float(cam_info[7])
    # camera intrinsic matrix
    K_mat = np.array([[f_x, 0, c_x],
                      [0, f_y, c_y],
                      [0,   0,  1]])

    projected_fruits_bboxes_with_hist_3D_pos_dict = project_landmarks_add_unseen_observation(current_frame_idx=50, landmark_horizon=[-30,-2], img_w_h = [1236, 1648], fruit_3D_pos_bboxes_hist_dict = fruit_3D_pos_bboxes_hist_dict, rot_trans_dict= rot_trans_dict, K_mat = K_mat)
