import cv2
import numpy as np
from skimage.color import rgb2gray
from skimage.transform import (hough_line, hough_line_peaks,
                               probabilistic_hough_line)
from skimage.feature import canny
import matplotlib.pyplot as plt
from matplotlib import cm
import os
from enum import Enum
import argparse

from imageai.Detection import ObjectDetection

import binascii
import struct

import scipy
import scipy.misc
import scipy.cluster
import colorsys


NUM_CLUSTERS_TEST = 10
NUM_CLUSTERS_TEMPLATE = 3
COLOR_OFFSET = 10
BIN_COLORS = []

CHAIR_X_OFFSET = 200
DOOR_LINE_ANGLE_EPS = 0.3


TEST_ANS_FILE = "test_ans.txt"

class Mode(Enum):
   PREPARE = 1
   TEST = 2
   AI = 3

MODE = Mode.TEST

execution_path = os.getcwd()

def chair_ai(detector, filename):

    detections, extracted_images = detector.detectObjectsFromImage(input_image=filename, output_image_path=os.path.join(execution_path , "imagenew.jpg"), extract_detected_objects=True)

    for eachObject in detections:
        print(eachObject["name"] , " : " , eachObject["percentage_probability"] )
        if eachObject["name"] == "chair":
            #print(eachObject["box_points"])
            return eachObject["box_points"][2] - eachObject["box_points"][0]
    return -1


def get_bin_colors(filename):
    im = cv2.imread(filename)
    ar = np.asarray(im)
    shape = ar.shape
    ar = ar.reshape(np.product(shape[:2]), shape[2]).astype(float)

    print('finding clusters')
    codes, dist = scipy.cluster.vq.kmeans(ar, NUM_CLUSTERS_TEMPLATE)
    print('cluster centres:\n', codes)

    vecs, dist = scipy.cluster.vq.vq(ar, codes)         # assign codes
    counts, bins = np.histogram(vecs, len(codes))    # count occurrences

    index_max = np.argmax(counts)
    codes = np.delete(codes, index_max, 0) # delete background color
    dif = codes[1] - codes[0]
    all_less = True
    for i in range(0,3):
        if dif[i] < 0:
            all_less = False
            break
    if not all_less:
        tmp = 0
        for i in range(0,3):
            tmp=codes[0][i]
            codes[0][i] = codes[1][i]
            codes[1][i] = tmp

    print('chair cluster centres:\n', codes)

    return codes

def clusterization(img):
    ar = np.asarray(img)
    shape = ar.shape
    ar = ar.reshape(np.product(shape[:2]), shape[2]).astype(float)

    print('finding clusters')
    codes, dist = scipy.cluster.vq.kmeans(ar, NUM_CLUSTERS_TEST)
    print('cluster centres:\n', codes)

    vecs, dist = scipy.cluster.vq.vq(ar, codes)         # assign codes
    counts, bins = np.histogram(vecs, len(codes))    # count occurrences

    index_max = np.argmax(counts)                    # find most frequent
    peak = codes[index_max]
    colour = binascii.hexlify(bytearray(int(c) for c in peak)).decode('ascii')
    #print('most frequent is %s (#%s)' % (peak, colour))


    c = ar.copy()
    for i, code in enumerate(codes):
        c[scipy.r_[np.where(vecs==i)],:] = code
    clustered = c.reshape(*shape).astype(np.uint8)
    print('end clustered image')
    return clustered


def chair_width(img):
    #img = cv2.resize(img, (int(312*1.7), int(416*1.7)))
    clustered = clusterization(img)
    #to_show = cv2.resize(clustered, (int(312*1.7), int(416*1.7)))   
    #cv2.imshow("e1", to_show)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()


    #136.82386855 147.75381954  14.02565581
    #79.41322415  91.13273038   8.65925202

    #31,100,50
    #70,255,255
    lower_green = np.array(BIN_COLORS[0], dtype=np.uint8)
    upper_green = np.array(BIN_COLORS[1], dtype=np.uint8)
    

   # print(lower_green, upper_green)
    hsv_lower = colorsys.rgb_to_hsv(lower_green[0]/255., lower_green[1]/255., lower_green[2]/255.)
    hsv_upper = colorsys.rgb_to_hsv(upper_green[0]/255., upper_green[1]/255., upper_green[2]/255.)
    

    hsv_lower = np.array([int(hsv_lower[0]*100), int(hsv_lower[1]*100), int(hsv_lower[2]*100)])
    hsv_upper = np.array([int(hsv_upper[0]*100), int(hsv_upper[1]*100), int(hsv_upper[2]*100)])
    for i in range(0,3):
        hsv_upper[i] = min(100, hsv_upper[i]+COLOR_OFFSET)
        hsv_lower[i] = max(0, hsv_lower[i]-COLOR_OFFSET)

   # print(hsv_lower, hsv_upper)

    image_hsv = cv2.cvtColor(clustered, cv2.COLOR_BGR2HSV)  
    only_template_color = cv2.inRange(image_hsv, hsv_lower, hsv_upper)

    #to_show = cv2.resize(only_template_color, (int(312*1.7), int(416*1.7)))   
    #cv2.imshow("e2", to_show)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    vertical_indices = np.where(np.any(only_template_color, axis=1))[0]
    top, bottom = vertical_indices[0], vertical_indices[-1]

    horizontal_indices = np.where(np.any(only_template_color, axis=0))[0]
    left_idx=0
    right_idx=-1
    left = horizontal_indices[left_idx]
    right = horizontal_indices[right_idx]
    while (left < CHAIR_X_OFFSET):
        left = horizontal_indices[left_idx]
        left_idx += 1

    while (right > only_template_color.shape[1]-CHAIR_X_OFFSET):
        right = horizontal_indices[right_idx]
        right_idx -= 1
    

    print(f"Top: {top}, bottom: {bottom}")
    print(f"Left: {left}, right: {right}")

    #from matplotlib.patches import Rectangle

    #f, ax = plt.subplots(1, 1)
    #ax.imshow(only_template_color)

    corner = (left, top)
    height = bottom - top
    width = right - left
    return width
    #ax.add_patch(Rectangle(corner, width, height, linewidth=5, edgecolor='b', facecolor='none'))
    #plt.savefig("sas", bbox_inches='tight')
    #cv2.imwrite('1.jpg', only_template_color)


def show_hough_transform(image, filename):
    h, theta, d = hough_line(canny(image)) # вычисляем преобразование Хаффа от границ изображения

    fig, ax = plt.subplots(1, 3, figsize=(15, 6))

    ax[0].imshow(image, cmap=cm.gray)
    ax[0].set_title('Input image')
    ax[0].set_axis_off()

    ax[1].imshow(np.log(1 + h),
                 extent=[np.rad2deg(theta[-1]), np.rad2deg(theta[0]), d[-1], d[0]],
                 cmap='gray', aspect=1/20)
    ax[1].set_title('Hough transform')
    ax[1].set_xlabel('Angles (degrees)')
    ax[1].set_ylabel('Distance (pixels)')

    ax[2].imshow(image, cmap=cm.gray)
    for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
        y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
        y1 = (dist - image.shape[1] * np.cos(angle)) / np.sin(angle)
        ax[2].plot((0, image.shape[1]), (y0, y1), '-r')
    ax[2].set_xlim((0, image.shape[1]))
    ax[2].set_ylim((image.shape[0], 0))
    ax[2].set_axis_off()
    ax[2].set_title('Detected lines')

    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close(fig)

def width_door_hough(image):
    low_dark = (0, 0, 0)
    high_dark = (50, 50, 50)
    only_dark = cv2.inRange(img, low_dark, high_dark)
    dists_left = []
    dist_right = []
    middle = image.shape[1]/2

    if cv2.waitKey(0) & 0xFF == ord('q'):  
       cv2.destroyAllWindows() 


    h, theta, d = hough_line(canny(only_dark))
    for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
        if angle > -DOOR_LINE_ANGLE_EPS and angle < DOOR_LINE_ANGLE_EPS:
            if dist < middle:
                dists_left.append(dist)
            if dist > middle:
                dist_right.append(dist)

    if len(dist_right) >= 1 and len(dist_right) >=1:
        return min(dist_right) - max(dists_left)
    else:
        return -1


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--mode", required=False,
     help="execution mode: 1 - PREPARE, 2 - TEST, 3 - AI detection")
    ap.add_argument("-s", "--dir", required=False,
     help="path to the input images dir")
    ap.add_argument("-f", "--test_file", required=False,
     help="path to the input test answers file")
    ap.add_argument("-t", "--template_dir", required=False,
     help="path to the color templates directory and template file")
    args = vars(ap.parse_args())

    MODE = Mode(Mode.TEST if args['mode'] is None else int(args['mode']))
    directory = ("DataSetV2" if args['dir'] is None else args['dir'])
    TEST_ANS_FILE = "test_ans.txt" if args['test_file'] is None else args['test_file']
    template_file = "template\\color_template_1_nice_back.jpg" if args['template_dir'] is None else args['template_dir']
    
    if MODE == Mode.PREPARE:
        f = open(TEST_ANS_FILE, "w")

    if MODE == Mode.AI:
        detector = ObjectDetection()
        detector.setModelTypeAsRetinaNet()
        detector.setModelPath( os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5"))
        detector.loadModel()

    if MODE == Mode.TEST or MODE == Mode.AI:
        test_ans = {}
        with open(TEST_ANS_FILE) as f:
            for line in f:
                (key, val) = line.split(" ")
                test_ans[key] = val

    if MODE == MODE.TEST:
        BIN_COLORS = get_bin_colors(template_file)

    total = 0
    errors = 0

    for file in os.listdir(directory):
        filename = os.fsdecode(file)

        if filename.endswith(".jpg"):
            total += 1
            print(MODE)

            if MODE == Mode.PREPARE:
               
                f.write(filename+"\n")

            if MODE == Mode.TEST:
                print(os.path.join(directory, filename, ""))

                img = cv2.imread(os.path.join(directory, filename))
                #img = rgb2gray(cv2.imread("Datas\\" + filename))
                #img = cv2.resize(img, (int(img.shape[1] / 2), int(img.shape[0] / 2)))
                door = width_door_hough(img)
                chair = chair_width(img)

                if (door == -1 or chair == -1):
                    result = '2'
                    print('result for',filename,'is', result, "expected", test_ans[filename])
                    if int(test_ans[filename]) != int(result):
                        errors += 1
                else:
                    result = '1' if (door < chair) else '0'
                    print('Door wight: ', door)
                    print('Chair width: ', chair)
                    print('result for',filename,'is', result, "expected", test_ans[filename])
                    if int(test_ans[filename]) != int(result):
                        errors += 1

            if MODE == Mode.AI:
                print("errors", errors, "total", total)
                print(os.path.join(directory, filename))
                img = cv2.imread(os.path.join(directory, filename))
                door = width_door_hough(img)
                chair = chair_ai(detector, os.path.join(directory, filename))

                if (door == -1 or chair == -1):
                    result = '2'
                    print('result for',filename,'is', result, "expected", test_ans[filename])
                    if int(test_ans[filename]) != int(result):
                        errors += 1
                else:
                    result = '1' if (door < chair) else '0'
                    print('Door wight: ', door)
                    print('Chair width: ', chair)
                    print('result for',filename,'is', result, "expected", test_ans[filename])
                    if int(test_ans[filename]) != int(result):
                        errors += 1
                
    if MODE == Mode.PREPARE:
        f.close()

    if MODE == Mode.TEST or MODE == Mode.AI:
        print(total, errors)
        print("accuracy: %f" % (float(total-errors) / float(total)))

    cv2.waitKey(0)
    cv2.destroyAllWindows()