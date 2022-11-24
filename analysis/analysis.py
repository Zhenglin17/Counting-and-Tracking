import numpy as np
import caffe
import pdb
from scipy import misc
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import glob
from PIL import Image
import cPickle as pickle

caffe_root = '../../../'
fruit_dir = caffe_root + 'Ag/dataset/apple/green/fast_led/'
numpy_save_directory = caffe_root + 'Ag/dataset/apple/green/fast_led/scores/'
pickle_save_location = caffe_root + 'Ag/code/python/data.p'


def load_label(idx):
    caffe_root = '../../../'
    frame = idx.replace(numpy_save_directory, "")
    frame = frame.replace('.npy', "")
    """
    Load label image as 1 x height x width integer array of label indices.
    The leading singleton dimension is required by the loss.
    """
    fruit_dir = caffe_root + 'Ag/dataset/apple/green/fast_led/'
    im = Image.open('{}labels/{}_pos.png'.format(fruit_dir, frame))
    # im = Image.open(glob.glob("{}sandeep_labels/*I{}.jpg.png".format(self.fruit_dir, idx))[0])
    # im = misc.imread(glob.glob("{}sandeep_labels/*I{}.jpg.png".format(self.fruit_dir, idx))[0])
    label = np.array(im, dtype=np.uint8)
    label = label / 255  # convert 255 to 1
    # label = label==0
    label = np.uint8(label)
    label = label[np.newaxis, ...]
    return label


def generate_datapoints(min_thresh=0, max_thresh=1, step_thresh=0.01):
    # caffe_root = '../../../'
    score_list = []
    label_list = []
    # numpy_save_file = caffe_root + 'Ag/dataset/apple/green/fast_led/scores/' + str(image_frame) + '.npy'
    for filename in glob.glob(numpy_save_directory + '*.npy'):
        score_list.append(np.load(filename))
        ind_label = load_label(filename)
        label_list.append(ind_label[0])

    # vstack them
    score = np.vstack(score_list)
    label = np.vstack(label_list)
    # score = np.load(numpy_save_file)
    # label = load_label(image_frame)
    # label = label[0]
    total_pixels = np.shape(label)[0] * np.shape(label)[1]
    num_steps = int(max_thresh / step_thresh)
    datapoints = []
    threshold = min_thresh
    for i in range(num_steps):
        thresholded_score = np.uint8(score > threshold)
        true_positive = np.sum(
            np.logical_and(np.uint8(thresholded_score > threshold),
                           label == 1)) / np.float64(np.sum(label == 1))
        false_positive = np.sum(
            np.logical_and(np.uint8(thresholded_score > threshold),
                           label == 0)) / np.float64(np.sum(label == 0))
        true_negative = np.sum(
            np.logical_and(np.uint8(thresholded_score < threshold),
                           label == 0)) / np.float64(np.sum(label == 0))
        false_negative = np.sum(
            np.logical_and(np.uint8(thresholded_score < threshold),
                           label == 1)) / np.float64(np.sum(label == 1))

        pixel_accuracy = np.float64((np.sum(
            np.logical_and(np.uint8(thresholded_score > threshold),
                           label == 1)) + np.sum(
            np.logical_and(np.uint8(thresholded_score < threshold),
                           label == 0)))) / total_pixels
        pos_pixel_accuracy = (np.sum(
            np.logical_and(np.uint8(thresholded_score > threshold),
                           label == 1))) / total_pixels
        neg_pixel_accuracy = (np.sum(
            np.logical_and(np.uint8(thresholded_score < threshold),
                           label == 0))) / total_pixels

        perc_pos = np.float64(np.sum(label == 1)) / total_pixels
        perc_neg = np.float64(np.sum(label == 0)) / total_pixels
        mean_accuracy = 0.5 * (true_positive + true_negative)

        IU_positive = np.sum(
            np.logical_and(np.uint8(thresholded_score > threshold),
                           label == 1)) / (
                      np.float64(np.sum(label == 1)) + np.sum(np.logical_and(
                          np.uint8(thresholded_score > threshold), label == 0)))
        IU_negative = np.sum(
            np.logical_and(np.uint8(thresholded_score < threshold),
                           label == 0)) / (
                      np.float64(np.sum(label == 0)) + np.sum(np.logical_and(
                          np.uint8(thresholded_score < threshold), label == 1)))
        mean_IU = (IU_positive + IU_negative) / 2
        frequency_IU = ((np.float64(np.sum(label == 1)) * IU_positive) + (
        np.float64(np.sum(label == 0)) * IU_negative)) / total_pixels

        datapoints.append(
            [threshold, false_positive, true_positive, false_negative,
             true_negative, pixel_accuracy, mean_accuracy, mean_IU,
             frequency_IU])
        threshold += step_thresh
    pickle.dump(datapoints, open(pickle_save_location, 'wb'))
    return datapoints


def plot_graphs(datapoints):
    fp = []
    tp = []
    pixel_accuracy = []
    mean_accuracy = []
    mean_IU = []
    frequency_weighted_IU = []
    x_axis = []

    for point in datapoints:
        x_axis.append(point[0])
        fp.append(point[1])
        tp.append(point[2])
        pixel_accuracy.append(point[5])
        mean_accuracy.append(point[6])
        mean_IU.append(point[7])
        frequency_weighted_IU.append(point[8])

    plt.subplot(2, 3, 1)
    plt.title("ROC")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.plot(fp, tp, 'ro')
    plt.subplot(2, 3, 2)
    plt.title("Pixel Accuracy")
    plt.xlabel("Threshold")
    plt.ylabel("Percentage")
    plt.plot(x_axis, pixel_accuracy, 'ro')
    plt.subplot(2, 3, 3)
    plt.title("Mean Accuracy")
    plt.xlabel("Threshold")
    plt.ylabel("Percentage")
    plt.plot(x_axis, mean_accuracy, 'ro')
    plt.subplot(2, 3, 4)
    plt.title("Mean IU")
    plt.xlabel("Threshold")
    plt.ylabel("Percentage")
    plt.plot(x_axis, mean_IU, 'ro')
    plt.subplot(2, 3, 5)
    plt.title("Frequency IU")
    plt.xlabel("Threshold")
    plt.ylabel("Percentage")
    plt.plot(x_axis, frequency_weighted_IU, 'ro')
    # plt.axis([0, 1, 0, 1])
    plt.show()


"""
def plot_error(datapoints):
   pixel_accuracy = []
   mean_accuracy = []
   mean_IU = []
   frequency_weighted_IU = []
   x_axis = []
   for point in datapoints:
       x_axis.append(point[0])
       pixel_accuracy.append(point[5])
       mean_accuracy.append(point[6])
       mean_IU.append(point[7])
       frequency_weighted_IU.append(point[8])
   #for point in datapoints:
   #    x_axis.append(point[0])
   #    fp.append(point[0])
   #    fn.append(point[1])
       #error.append(point[5]+point[6])
   plt.plot(x_axis, pixel_accuracy, 'ro')
   plt.plot(x_axis, fp, 'go')
   plt.plot(x_axis, fn, 'bo')
   plt.axis([0, 1, 0, 0.2])
   red_patch = mpatches.Patch(color='red', label='Total error')
   green_patch = mpatches.Patch(color='green', label='Positive error')
   blue_patch = mpatches.Patch(color='blue', label='Negative error')
   plt.legend(handles=[red_patch, green_patch, blue_patch])
   plt.show()
   
"""
