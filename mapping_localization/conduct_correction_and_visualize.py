from __future__ import (print_function, division, absolute_import)
import pickle
import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def conduct_correction_and_visualize(dis_rejection = False, double_tracks = False, display_semantic_mapping = True, display_distance_histogram = True):

    with open('generated_docs_this_run/pos_area_fridx_of_fruits.pkl', 'rb') as input:
        fruits_pos_area_list = pickle.load(input)
    fruits_pos_area_arr = np.array(fruits_pos_area_list)

    fruits_cam_frame = np.load('generated_docs_this_run/pos_cam_frame_area_fridx_of_fruits.npy')
    zs_cam_frame = fruits_cam_frame[:,2]
    zs_mean = np.mean(zs_cam_frame)
    sizes = fruits_cam_frame[:,3]
    fridx = fruits_cam_frame[:,4]


    xs = fruits_pos_area_arr[:,0]
    ys = fruits_pos_area_arr[:,1]
    zs = fruits_pos_area_arr[:,2]
    num_pts = xs.shape[0]


    # Plot area in 3D space (fuse depth data) distribution
    fruits_pos_area_id_arr_camera_frame = np.load('generated_docs_this_run/pos_cam_frame_area_fridx_of_fruits.npy')
    projected_area = fruits_pos_area_id_arr_camera_frame[:,3]
    projected_area /= projected_area.mean()

    img_pos = np.load('generated_docs_this_run/img_id_position.npy')
    x_imgs = img_pos[:,1]
    y_imgs = img_pos[:,2]
    z_imgs = img_pos[:,3]


    area = fruits_pos_area_arr[:,3]
    xs_range = xs.max() - xs.min()
    ys_range = ys.max() - ys.min()
    zs_range = zs.max() - zs.min()
    over_count_fruits = 0

    xs_c = xs
    ys_c = ys
    zs_c = zs

    #TODO: double count removal
    mean_depth = zs_cam_frame.mean()
    double_counted_list = []
    over_count_fruits_prev = -1
    for idx_pt in range(num_pts):
        for idx_pt_2 in range(idx_pt+1, num_pts):
            # TODO: NEW Correction: dis was wrong in previous version of code, where dis = np.linalg.norm(idx_pt-idx_pt_2)
            dis = np.linalg.norm(xs[idx_pt]-xs[idx_pt_2]) + np.linalg.norm(ys[idx_pt]-ys[idx_pt_2]) + np.linalg.norm(zs[idx_pt]-zs[idx_pt_2]) / 2.0

            if sizes[idx_pt] > 2* sizes[idx_pt_2] or sizes[idx_pt_2] > 2* sizes[idx_pt]:
                continue
            dis_size_ratio = dis / np.sqrt(sizes[idx_pt]+sizes[idx_pt_2])
            # TODO: normalize with depth (equal to normal size with depth ^2)
            normed_dis_size_ratio = dis_size_ratio / mean_depth
            if normed_dis_size_ratio<0.2:
                over_count_fruits += 1
                print('frame#:',fridx[idx_pt], '  double counted fruits found' + str(over_count_fruits) + '\n')
                xs[idx_pt] = 0
                ys[idx_pt] = 0
                zs[idx_pt] = 0
        if (fridx[idx_pt] != fridx[idx_pt-1] or idx_pt == 0) and (over_count_fruits != over_count_fruits_prev):
            double_counted_list.append([fridx[idx_pt],over_count_fruits])
            over_count_fruits_prev = over_count_fruits
    np.savetxt('generated_docs_this_run/double_counted.txt', double_counted_list, '%d', header= 'Frame#, accumulative double count')
    #
    # #TODO: distance outlier removal
    distance_out_liers = 0
    dist_outliers_list = []
    for idx_dis in range(num_pts):
        if zs_cam_frame[idx_dis] > 1.5*zs_mean:
            distance_out_liers += 1
            zs[idx_dis] = 0
            xs[idx_dis] = 0
            ys[idx_dis] = 0
            if fridx[idx_dis] != fridx[idx_dis - 1] or idx_dis == 0:
                print('frame# and size outliers', fridx[idx_dis], distance_out_liers)
            dist_outliers_list.append([fridx[idx_dis], distance_out_liers])
    np.savetxt('generated_docs_this_run/distance_outliers.txt', dist_outliers_list, '%d',
               header='Frame#, accumulative distance outliers')

    # TODO: SIZE outlier removal (in the end)
    size_outliers_list = []
    multi_fruit_list = []
    out_liers_size = 0
    multiple_fruit = 0
    num_projected_area = projected_area.shape[0]
    for idx_pro in range(num_projected_area):
        if xs[idx_pro]==0 and zs[idx_pro]==0 and ys[idx_pro]==0:
            continue

        # # TODO: take double fruit into consideration
        # if 4 > projected_area[idx_pro] > 2:
        #     multiple_fruit+= np.sqrt(projected_area[idx_pro])
        #     print('frame# and multiple fruit', fridx[idx_pro], multiple_fruit)
        #     multi_fruit_list.append([fridx[idx_pro], multiple_fruit])
        if projected_area[idx_pro] > 4 or projected_area[idx_pro] < 0.5:

            out_liers_size += 1

            # if fridx[idx_pro] != fridx[idx_pro - 1] or idx_pro == 0:
                # print('frame# and size outliers', fridx[idx_pro], out_liers_size)
            c = fridx[idx_pro]
            # if c==1 or c ==108 or c==118 or c == 186 or c==195 or c==308 or c==301 or c== 410 or c==407 or c==533 \
            #         or c==552 or c==563 or c== 602 or c==713 or c==705 or c==811 or c==799 or c==918 or c==904 or c==1009 or c==1001 or c==1090:
            #     print('frame# and size outliers', fridx[idx_pro], out_liers_size)
            size_outliers_list.append([fridx[idx_pro], out_liers_size])
    np.savetxt('generated_docs_this_run/size_outliers.txt', size_outliers_list, '%d',
               header='Frame#, accumulative size outliers')
    np.savetxt('generated_docs_this_run/multiple_fruit.txt', multi_fruit_list, '%.1f',
               header='Frame#, accumulative size outliers')

    def remove_size_outlier(projected_area):
        out_liers_size = 0
        num_projected_area = projected_area.shape[0]
        for idx_pro in range(num_projected_area):
            if projected_area[idx_pro] > 2 or projected_area[idx_pro] < 0.25:
                out_liers_size += 1
                print('size outliers', out_liers_size)


    ax = Axes3D(plt.gcf())

    # TODO: only for plot figure for paper, remember to delete this!!
    # xs[xs>-4]=0
    # x_imgs[x_imgs>-4]=0
    # ax.scatter(xs[0:1000], ys[0:1000], zs[0:1000], zdir='z', s=0.2, c='red', label = 'fruits')
    # axis_0_1 = np.array([0,1])
    # axis_0_0 = np.array([0,0])
    # ax.scatter(x_imgs, y_imgs, z_imgs, zdir='z', s = 5, marker='^', c = 'blue', label = 'projection centers')
    # ax.scatter(0, 0, 0, zdir='z', s = 40, c = 'black', label = 'world coordinate center')
    # ax.plot(axis_0_0, axis_0_0, axis_0_1*0.7,zdir='z', c ='green', label = 'world coordinate Z axis')
    # ax.plot(axis_0_0, axis_0_1*0.2, axis_0_0,zdir='z', c ='blue')
    # ax.plot(axis_0_1, axis_0_0, axis_0_0,zdir='z', c ='red')
    # ax.set_zlim(-4,8)
    # ax.grid('off')
    # plt.xlim(-10,-1)
    # plt.ylim(-3,3.5)


    # TODO:
    ax.scatter(xs, ys, zs, zdir='z', s=0.2, c='red', label = 'fruits')
    axis_0_1 = np.array([0,1])
    axis_0_0 = np.array([0,0])
    ax.scatter(x_imgs, y_imgs, z_imgs, zdir='z', s = 5, marker='^', c = 'blue', label = 'projection centers')
    ax.scatter(0, 0, 0, zdir='z', s = 80, c = 'black', label = 'world coordinate center')
    ax.plot(axis_0_0, axis_0_0, axis_0_1*0.7,zdir='z', c ='green', label = 'world coordinate Z axis')
    ax.plot(axis_0_0, axis_0_1*0.2, axis_0_0,zdir='z', c ='blue')
    ax.plot(axis_0_1, axis_0_0, axis_0_0,zdir='z', c ='red')
    ax.set_zlim(-8,8)
    ax.grid('off')
    plt.axis('off')
    plt.xlim(-10,10)
    plt.ylim(-3,3.5)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()

    plt.show(ax)


    # Plot area in image plane distribution
    # count
    # count = plt.hist(area, bins=area.shape[0])
    # #x, bins=None, range=None, density=None, weights=None, cumulative=False, bottom=None, histtype='bar', align='mid', orientation='vertical', rwidth=None, log=False, color=None, label=None, stacked=False, normed=None, hold=None, data=None, **kwargs)
    # plt.xlim(0,2500)
    # plt.xlabel('BBox area in pixels')
    # plt.ylabel('Count')
    # plt.title('Area in Image Plane (0-2500)')
    # plt.show(count)

    # probability
    # prob = plt.hist(area, normed=True, bins=30)
    # plt.xlabel('BBox area in pixels')
    # plt.ylabel('Probability')
    # plt.show(prob)





    # scatter plot
    # scatter1 = plt.scatter(np.arange(projected_area.shape[0]),projected_area,s = 1)
    # scatter1 = plt.scatter(projected_area,np.arange(projected_area.shape[0]),s = 1)
    # plt.xlim(0,3)
    # plt.xlabel('Fruit Size Divided by Mean')
    # plt.ylabel('Index of fruits')
    # plt.title('Relative Size Scatter(Majority)')# (0-2000)
    # plt.show(scatter1)

    scatter1 = plt.scatter(projected_area,np.arange(projected_area.shape[0]),s = 1, c='green')
    plt.xlabel('Fruit Size Divided by Mean')
    plt.ylabel('Index of fruits')
    plt.title('Relative Size Scatter')# (0-2000)
    plt.show(scatter1)

    # count
    count2 = plt.hist(projected_area, bins = 100,color = 'green')#bins=projected_area.shape[0]
    plt.plot([0.5,0.5],[0,200],'blue',label = 'Lower threshold')
    plt.plot([4,4],[0,200],'red',label = 'Upper threshold')
    # projected_area_outlier1 = projected_area[projected_area<0.5]
    # projected_area_outlier2 = projected_area[projected_area>2]
    # count2 = plt.hist(projected_area_outlier1, bins = 200,color = 'green')
    # plt.xlim(0,200)
    plt.xlabel('Fruit Size Divided by Mean')
    plt.ylabel('Count')
    plt.title('Relative Size') # (0-2000)
    plt.legend()
    plt.show(count2)


    # TODO: SIZE outlier removal
    # remove_size_outlier(projected_area)


    # count

    # count2 = plt.hist(projected_area, bins = 200)#bins=projected_area.shape[0]
    # plt.xlim(0.25,2)
    # plt.xlabel('Fruit Size Divided by Mean')
    # plt.ylabel('Count')
    # plt.title('Apples: Relative Size(After Removing Size Outliers)') # (0-2000)
    # plt.show(count2)

    # # prob
    # prob2 = plt.hist(projected_area, normed = True,bins=projected_area.shape[0]+1000,color = 'green')
    # plt.xlim(0,3000)
    # plt.xlabel('Relative Size of Fruits (0-3000)')
    # plt.ylabel('Count')
    # plt.title('Relative Size Probability')
    # plt.show(prob2)
    #
    # print('over')


if __name__=="__main__":
    conduct_correction_and_visualize()
