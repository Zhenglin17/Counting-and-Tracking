import cv2
import os
import glob
# only_two_frames = False
# if only_two_frames:

dataset = 'ACFR'
frames = []

if 'recon' not in dataset:
    if dataset == 'ACFR':
        image_dir = '/home/sam/Desktop/Fruit-count-dataset/ACFR_RCNN_resized/tracking_results'
        video_name = '/home/sam/Desktop/counting_video_acfr.avi'

    if dataset == 'apple':
        image_dir = 'C:\\Users\\UPenn Robotics LX\\Desktop\\Fruit-count-dataset\\apple_slow_flash_tracking_results\\finally_use_apple_tracking_result_age=5'
        video_name = 'C:\\Users\\UPenn Robotics LX\\Desktop\\counting_video_apple.avi'

    if dataset == 'orange':
        image_dir = 'C:\\Users\\UPenn Robotics LX\\Desktop\\Fruit-count-dataset\\oranges_tracking_results\\finally_use_orange_tracking_result_age=5'
        video_name = 'C:\\Users\\UPenn Robotics LX\\Desktop\\counting_video_orange.avi'

    if dataset == 'trunk':
        # image_dir = '/home/sam/Desktop/others/Fruit-count-dataset/trunk/trunk_tracking_results'
        image_dir = '/home/sam/Desktop/Fruit-count-dataset/apple/red/slow_flash/north/bag/track/frame1_trunk'
        video_name = '/home/sam/Desktop/counting_video_trunk2.avi'

    if dataset == 'trunk_walnut':
        # image_dir = '/home/sam/Desktop/others/Fruit-count-dataset/trunk/trunk_tracking_results'
        image_dir = '/home/sam/Desktop/Fruit-count-dataset/apple/red/slow_flash/north/bag/track/frame1_trunk_walnut'
        video_name = '/home/sam/Desktop/counting_video_trunk_walnut.avi'

    video_interval = 1
    frame_rate = 10
    x_resize = 1
    y_resize = 1
    for i in range(0,1500,video_interval):
        ind = '%04d' % i
        if os.path.isfile(image_dir + '/bgr' + ind + '.png'):
            frametemp = cv2.imread(image_dir + '/bgr' + ind + '.png')
            frametemp = cv2.resize(frametemp,(0,0), fx=x_resize, fy=y_resize)
            h,w,_ = frametemp.shape
            frame = frametemp[-w:,:,:]
            frames.append(frame)
            print ind
        else:
            print (image_dir + '/bgr' + ind + '.png' + ' is not found')
            break

height, width, layers = frame.shape

fourcc = cv2.VideoWriter_fourcc(*'MJPG')
video = cv2.VideoWriter(video_name, fourcc = fourcc, fps = frame_rate, frameSize = (width,height), isColor=True)

for frame in frames:
    video.write(frame)

cv2.destroyAllWindows()
video.release()
print('video saved!')
