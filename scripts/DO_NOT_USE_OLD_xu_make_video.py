import cv2
import os
only_two_frames = True
if only_two_frames:
    image_dir = 'C:\\Users\\UPenn Robotics LX\\Desktop\\Fruit-count-dataset\\apple_slow_flash_tracking_results\\frame1_min_age=5_final_use'
    video_name = 'C:\\Users\\UPenn Robotics LX\\Desktop\\counting_video_1.avi'
else:
    image_dir = 'C:\\Users\\UPenn Robotics LX\\Desktop\\Fruit-count-dataset\\apple_slow_flash_tracking_results\\\frame1_min_age=5_final_use'
    video_name = 'C:\\Users\\UPenn Robotics LX\\Desktop\\counting_video.avi'
frames = []
for i in range(1,1100):
    ind = '%04d' % i
    if os.path.isfile(image_dir + '\\disp' + ind + '.png'):
        frame = cv2.imread(image_dir + '\\disp' + ind + '.png')
        frames.append(frame)
    else:
        print (image_dir + '\\disp' + ind + '.png' + ' is not found')

height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, fourcc = -1, fps = 10, frameSize = (width,height), isColor=True)

for frame in frames:
    video.write(frame)

cv2.destroyAllWindows()
video.release()
