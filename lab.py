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

execution_path = os.getcwd()




CHAIR_X_OFFSET = 200
DOOR_LINE_ANGLE_EPS = 0.3

TEST_ANS_FILE = "test_ans.txt"

class Mode(Enum):
   PREPARE = 1
   TEST = 2
   AI = 3

MODE = Mode.TEST



def chair_ai(detector, filename):

    detections, extracted_images = detector.detectObjectsFromImage(input_image=filename, output_image_path=os.path.join(execution_path , "imagenew.jpg"), extract_detected_objects=True)

    for eachObject in detections:
        print(eachObject["name"] , " : " , eachObject["percentage_probability"] )
        if eachObject["name"] == "chair":
            #print(eachObject["box_points"])
            return eachObject["box_points"][2] - eachObject["box_points"][0]
    return -1




def chair_width(img):
    #img = cv2.imread("DataSetV2\\IMG_20200306_193234.jpg")
    lower_green = np.array([31,100,50], dtype=np.uint8)
    upper_green = np.array([70,255,255], dtype=np.uint8)
    image_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  
    only_green = cv2.inRange(image_hsv, lower_green, upper_green)


    vertical_indices = np.where(np.any(only_green, axis=1))[0]
    top, bottom = vertical_indices[0], vertical_indices[-1]

    horizontal_indices = np.where(np.any(only_green, axis=0))[0]
    left_idx=0
    right_idx=-1
    left = horizontal_indices[left_idx]
    right = horizontal_indices[right_idx]
    while (left < CHAIR_X_OFFSET):
        left = horizontal_indices[left_idx]
        left_idx += 1

    while (right > only_green.shape[1]-CHAIR_X_OFFSET):
        right = horizontal_indices[right_idx]
        right_idx -= 1
    

    print(f"Top: {top}, bottom: {bottom}")
    print(f"Left: {left}, right: {right}")

    #from matplotlib.patches import Rectangle

    #f, ax = plt.subplots(1, 1)
    #ax.imshow(only_green)

    corner = (left, top)
    height = bottom - top
    width = right - left
    return width
    #ax.add_patch(Rectangle(corner, width, height, linewidth=5, edgecolor='b', facecolor='none'))
    #plt.savefig("sas", bbox_inches='tight')
    #cv2.imwrite('1.jpg', only_green)


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
    args = vars(ap.parse_args())

    MODE = Mode(Mode.TEST if args['mode'] is None else int(args['mode']))
    directory = ("DataSetV2" if args['dir'] is None else args['dir'])
    TEST_ANS_FILE = "test_ans.txt" if args['test_file'] is None else args['test_file']
    print(directory)
    
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