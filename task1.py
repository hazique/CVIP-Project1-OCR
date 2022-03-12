"""
Character Detection

The goal of this task is to implement an optical character recognition system consisting of Enrollment, Detection and Recognition sub tasks

Please complete all the functions that are labelled with '# TODO'. When implementing the functions,
comment the lines 'raise NotImplementedError' instead of deleting them.

Do NOT modify the code provided.
Please follow the guidelines mentioned in the project1.pdf
Do NOT import any library (function, module, etc.).
"""


import argparse
import json
import os
import glob
import cv2
import numpy as np
import connected_component_labeller
import math

def get_binary_inv_image(img):
    """ Args:
        img: numpy image array with original pixel values
    Returns:
        bw: numpy image array with data pixel values as 255 and rest 0
    """
    ret, bw = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    return bw


def get_img_features(bw_1):
    """ Args:
        bw1: image patch with data pixel value as 1 and rest 0
    Returns:
        a tuple of feature descriptors calculated below
    """

    if bw_1.shape[0] % 2 != 0:
        horizontal_padding = np.zeros((1, bw_1.shape[1]))
        bw_1 = np.append(bw_1, horizontal_padding, 0)

    if bw_1.shape[1] % 2 != 0:
        vertical_padding = np.zeros((bw_1.shape[0], 1))
        bw_1 = np.append(bw_1, vertical_padding, 1)

    quadrants = [M for SubA in np.split(bw_1, 2, axis=0)
                 for M in np.split(SubA, 2, axis=1)]

    pixel_sum = np.sum(bw_1)

    f1 = np.sum(quadrants[0])/pixel_sum
    f2 = np.sum(quadrants[1])/pixel_sum
    f3 = np.sum(quadrants[2])/pixel_sum
    f4 = np.sum(quadrants[3])/pixel_sum

    f5 = f1 + f2
    f6 = f2 + f3
    f7 = f3 + f4
    f8 = f1 + f4
    f9 = f2 + f4
    f10 = f1 + f3

    # Corners not needed as it makes selecting a maximum threshold distance ambiguous
    # corners = cv2.goodFeaturesToTrack(bw_255, maxCorners=50, qualityLevel=0.4, minDistance=2)
    # f11 = len(corners) if type(corners) == type(np.array([])) else 0

    f12 = np.sum(bw_1) / (bw_1.shape[0] * bw_1.shape[1])

    return (f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f12)


def get_bbox_coordinates(img, label, enroll=True):
    """ Args:
        img: labelled test image where each character pixel is replaced by
            the label assigned to it
    Returns:
        an array of starting X coordinate, starting Y coordinate, width and height of character
    """
    startX = startY = endX = endY = 0
    for y, row in enumerate(img):
        for x, pixel in enumerate(row):
            if pixel == label:
                if startX == 0 and startY == 0:
                    startX = x
                    startY = y
                    endX = x
                    endY = y   
                if x <= startX:
                    startX = x
                if x >= endX:
                    endX = x
                if y > endY:
                    endY = y
    width = endX - startX + 1
    height = endY - startY + 1
    if not enroll:
        return {
            "bbox": [startX, startY, width, height],
            "name": "unknown"
        }    
    else:
        return [startX, startY, width, height]

def ssd(f1, f2):
    """ 
    Args:
        f1: feature descriptor of the detected character
        f2: feature descriptor of the enrolled character
    Returns:
        Calulates and returns the Euclidean distance between the descriptors
    """
    return math.sqrt(np.sum((f1 - f2)**2))


# The following function is only for dev time verification
def verify_results(results):
    
    recognized = list(filter(lambda i : (i["name"] != "UNKNOWN"), results))

    for character in recognized:
        print("Recognized character: ", character["name"])
        show_image(character["image"])



def read_image(img_path, show=False):
    """Reads an image into memory as a grayscale array.
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if show:
        show_image(img)

    return img

def show_image(img, delay=1000):
    """Shows an image.
    """
    cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('image', img)
    cv2.waitKey(delay)
    cv2.destroyAllWindows()


def showImage(img):
    cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(description="cse 473/573 project 1.")
    parser.add_argument(
        "--test_img", type=str, default="./data/test_img.jpg",
        help="path to the image used for character detection (do not change this arg)")
    parser.add_argument(
        "--character_folder_path", type=str, default="./data/characters",
        help="path to the characters folder")
    parser.add_argument(
        "--result_saving_directory", dest="rs_directory", type=str, default="./",
        help="directory to which results are saved (do not change this arg)")
    args = parser.parse_args()
    return args

def ocr(test_img, characters):
    """Step 1 : Enroll a set of characters. Also, you may store features in an intermediate file.
       Step 2 : Use connected component labeling to detect various characters in an test_img.
       Step 3 : Taking each of the character detected from previous step,
         and your features for each of the enrolled characters, you are required to a recognition or matching.

    Args:
        test_img : image that contains character to be detected.
        characters_list: list of characters along with name for each character.

    Returns:
    a nested list, where each element is a dictionary with {"bbox" : (x(int), y (int), w (int), h (int)), "name" : (string)},
        x: row that the character appears (starts from 0).
        y: column that the character appears (starts from 0).
        w: width of the detected character.
        h: height of the detected character.
        name: name of character provided or "UNKNOWN".
        Note : the order of detected characters should follow english text reading pattern, i.e.,
            list should start from top left, then move from left to right. After finishing the first line, go to the next line and continue.
        
    """
    # TODO Add your code here. Do not modify the return and input arguments

    character_features = enrollment(characters)

    bboxes = detection(test_img)

    return recognition(character_features, bboxes, test_img)

def enrollment(characters):
    """ Args:
        characters: list of dictionaries having character name
            and correspoding and numpy image array
    Returns:
        features: list of dictionaries having character name
            and corresponding feature descriptor
    """
    # TODO: Step 1 : Your Enrollment code should go here.
    character_features = dict()
    for char in characters:

        label, img = char

        bw_255 = get_binary_inv_image(img)
        bw_1 = np.where(bw_255 == 255, 1, 0)
        x, y, width, height = get_bbox_coordinates(bw_1, 1)
        bw_1 = bw_1[y : y + height, x : x+width]

        features = get_img_features(bw_1)
        # character_features.append(features)
        character_features[label] = features

    return character_features


def detection(test_img):
    """
    Use connected component labeling to detect various characters in an test_img.
    Args:
        test_img: input test image from which characters are to be detected
    Returns:
        list of boundary boxes for each character detected in test_img
    """
    test_img_bin_inv = get_binary_inv_image(test_img)

    labelled_img = connected_component_labeller.label(test_img_bin_inv)

    bboxes = []
    for label in np.unique(labelled_img):
        record = get_bbox_coordinates(labelled_img, label, enroll=False)
        bboxes.append(record)

    return bboxes[1:]


def recognition(character_features, bboxes, test_img):
    """ 
    Args:
        character_features: dictionary of feature descriptors of enrolled characters
        bboxes: boundary box parameters of characters already detected in test_img
    Returns:
        List of characters recognized from the input set
    """

    test_img_inv = get_binary_inv_image(test_img)

    results = []
    verifiables = []

    for bbox in bboxes:

        detected_img = None
        x, y, width, height = bbox["bbox"]
        patch_255 = test_img_inv[y : y + height, x : x + width]

        patch_1 = np.where(patch_255 == 255, 1, 0)
        patch_features = np.array(get_img_features(patch_1))

        ssd_dict = {}
        for key in character_features.keys():
            enrol_feature = np.array(character_features[key])

            sum_sqr = ssd(patch_features, enrol_feature)
            ssd_dict[key] = sum_sqr

        entry = list(filter(lambda key: (ssd_dict[key] == min(ssd_dict.values()) and min(ssd_dict.values()) < 0.08), ssd_dict))
        result = {
            "bbox": bbox["bbox"],
            "name": entry[0] if len(entry) > 0 else "UNKNOWN"
        }
        verifiable = {
            "bbox": bbox["bbox"],
            "name": entry[0] if len(entry) > 0 else "UNKNOWN",
            "image": patch_255,
            "distance": ssd_dict[entry[0]] if len(entry) > 0 else 0
        }
 
        results.append(result)
        verifiables.append(verifiable)

    # verify_results(verifiables)
    return results



def save_results(coordinates, rs_directory):
    """
    Donot modify this code
    """
    results = coordinates
    with open(os.path.join(rs_directory, 'results.json'), "w") as file:
        json.dump(results, file)


def main():
    args = parse_args()
    
    characters = []

    all_character_imgs = glob.glob(args.character_folder_path+ "/*")
    
    for each_character in all_character_imgs :
        character_name = "{}".format(os.path.split(each_character)[-1].split('.')[0])
        characters.append([character_name, read_image(each_character, show=False)])

    test_img = read_image(args.test_img)
    results = ocr(test_img, characters)

    save_results(results, args.rs_directory)


if __name__ == "__main__":
    main()
