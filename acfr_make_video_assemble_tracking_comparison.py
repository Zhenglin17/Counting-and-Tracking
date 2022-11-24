import cv2
import os
import glob
from itertools import izip
import numpy as np
# only_two_frames = False
# if only_two_frames:

# data_dir_pre = '/home/sam/Desktop/Fruit-count-dataset/ACFR_RCNN_GT/done'
data_dir_pre = '/home/sam/Desktop/Fruit-count-dataset/ACFR_RCNN_GT_HA1.2_age3/done'

image_dirs = sorted(glob.glob(data_dir_pre+'/*/landmark_counting_visulization'))
image_dirs2 = sorted(glob.glob(data_dir_pre+'/*/projection_results'))

# image_dir = '/home/sam/Desktop/Fruit-count-dataset/ACFR_RCNN_GT/done/*/tracking_results_masked_visulization'
# image_dir2 = '/home/sam/Desktop/Fruit-count-dataset/ACFR_RCNN_GT/done/*/landmark_counting_visulization'
# image_dir3 = '/home/sam/Desktop/Fruit-count-dataset/ACFR_RCNN_GT/done/*/projection_results'
# image_dir4 = '/home/sam/Desktop/Fruit-count-dataset/ACFR_RCNN_GT/done/*/target_tree_mask'
# video_name = '/home/sam/Desktop/counting_video.avi'
finished_num = 0
all_num = len(image_dirs)
for image_dir, image_dir2 in izip(image_dirs, image_dirs2):
    finished_num += 1
    print 'current directory:',image_dir,'\n'
    print finished_num,'out of',all_num, 'folders finished'
    frames = []
    if (image_dir.split('/'))[-2] != (image_dir2.split('/'))[-2]:
        print image_dir
        print image_dir2
        raise Exception('check image dirs, they should be the same!!!')

    if 'hard' in (image_dir.split('/'))[-2]:
        print('This is the hard case, use the raw 2D tracking output to make video')
        image_dir = os.path.join(data_dir_pre,image_dir.split('/')[-2], 'tracking_results_masked_visulization')


    video_name = os.path.join(data_dir_pre,image_dir.split('/')[-2], 'counting_video_' +image_dir.split('/')[-2] +  '.avi')
    video_name_old = os.path.join(data_dir_pre,image_dir.split('/')[-2], 'counting_video.avi')

    if os.path.exists(video_name):
        os.remove(video_name)
    if os.path.exists(video_name_old):
        os.remove(video_name_old)


    video_interval = 1
    frame_rate = 5
    x_resize = 1.0
    y_resize = 1.0
    for i in range(0,200,video_interval):
        ind = '%04d' % i
        ind2 = ind

        if os.path.isfile(image_dir + '/bgr' + ind + '.png'):
            if 'hard' in (image_dir.split('/'))[-2]:
                frame = cv2.imread(image_dir + '/bgr' + ind + '.png')
                frame = cv2.resize(frame, (0,0), fx=x_resize, fy=y_resize)
                frame_final = frame
                frames.append(frame)
                if int(ind) %10 ==0:
                    print 'current index:', ind
            else:
                frame = cv2.imread(image_dir + '/bgr' + ind + '.png')
                frame2_name = (image_dir2 + '/projection_visualization_for_frame' + ind2 + '.png')
                # frame2_name = (image_dir2 + '/bgr' + ind2 + '.png')
                if os.path.isfile(frame2_name):
                    frame2 = cv2.imread(frame2_name)

                else:
                    print (frame2_name + 'does not exists, continue to the next image!')
                    continue
                # frame3 = cv2.imread(image_dir3 + '/projection_visualization_for_frame' + ind2 + '.png')
                # if frame3 is None:
                #     continue
                frame = cv2.resize(frame, (0,0), fx=x_resize, fy=y_resize)
                frame2 = cv2.resize(frame2, (0,0), fx=x_resize, fy=y_resize)
                # frame3 = cv2.resize(frame3,(0,0), fx=x_resize, fy=y_resize)
                rows,cols,_ = frame.shape
                frame_final = np.ones((rows,2*cols+20,3))*128
                frame_final[:, :cols, :] = frame
                frame_final[:, cols+20:2*cols+20, :] = frame2
                # frame_final[:, 2*cols+40:, :] = frame3
                frame_final = np.uint8(frame_final)

                # cv2.imshow('img',frame_final)
                # cv2.waitKey(1000000)
                # cv2.destroyAllWindows()

                frames.append(frame_final)
                if int(ind) %10 ==0:
                    print 'current index:', ind
        else:
            # print (image_dir + '/bgr' + ind + '.png' + ' is not found')
            continue



    height, width, layers = frame_final.shape

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    video = cv2.VideoWriter(video_name, fourcc = fourcc, fps = frame_rate, frameSize = (width,height), isColor=True)

    print('the last few frames are not good, here taking them out!')
    if len(frames) < 20:
        continue
    for frame in frames[:-15]:
        video.write(frame)

    cv2.destroyAllWindows()
    video.release()
    print('video saved!')
