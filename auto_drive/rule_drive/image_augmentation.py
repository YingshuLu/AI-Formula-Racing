import os
import sys
from PIL import Image
from PIL import ImageEnhance
import random
import cv2, numpy as np

base_image_folder = sys.argv[1]
print(base_image_folder)

def adjust_brightness(base_img, base_folder, base_filename):       
    for factor in [0.97,0.98,0.99]:
        image = base_img.convert('RGBA')
        image = ImageEnhance.Brightness(image).enhance(factor)
        image = image.convert('RGB')

        new_file_name = os.path.join(
            base_folder, '{}_{}.jpg'.format(os.path.splitext(base_filename)[0], factor))
        image.save(new_file_name)

def rotate_image(base_img, base_folder, base_filename):       
    for factor in [5,-2,2,5]:
        image = base_img.rotate(factor)
        
        image = image.convert('RGB')
        new_file_name = os.path.join(
            base_folder, '{}_{}.jpg'.format(os.path.splitext(base_filename)[0], factor))
        image.save(new_file_name)
        convert_black_to_red(new_file_name)

def convert_black_to_red(img_path):
    img = cv2.imread(img_path)
    b,g,r = cv2.split(img)

    blackmask = (r<150) & (b<150) & (g<150)
    r[blackmask] = 255
    img = cv2.merge((b, g, r))
    cv2.imwrite(img_path, img)


def resize_img(image, new_size_row=20, new_size_col=30):
    img = cv2.imread(image)
    # note: numpy arrays are (row, col)!
    img = cv2.resize(img,(new_size_col, new_size_row), interpolation=cv2.INTER_AREA)    
    cv2.imwrite(image, img)    

#adjust brightness
for sub_folder in [x for x in os.listdir(base_image_folder) if os.path.isdir(os.path.join(base_image_folder, x))]:
    print('to process sub folder {}'.format(sub_folder))
    folder_to_process = os.path.join(base_image_folder, sub_folder)
    for image_filename in [x for x in os.listdir(folder_to_process)]:
        print('to process image {}'.format(image_filename))
        image_path = os.path.join(folder_to_process, image_filename)
        base_img = Image.open(image_path)
        adjust_brightness(base_img, folder_to_process,
                        image_filename)

# rotate
for sub_folder in [x for x in os.listdir(base_image_folder) if os.path.isdir(os.path.join(base_image_folder, x))]:
    print('to process sub folder {}'.format(sub_folder))
    folder_to_process = os.path.join(base_image_folder, sub_folder)
    for image_filename in [x for x in os.listdir(folder_to_process)]:
        print('to process image {}'.format(image_filename))
        image_path = os.path.join(folder_to_process, image_filename)
        base_img = Image.open(image_path)
        rotate_image(base_img, folder_to_process,
                        image_filename)

# resize
for sub_folder in [x for x in os.listdir(base_image_folder) if os.path.isdir(os.path.join(base_image_folder, x))]:
    print('to process sub folder {}'.format(sub_folder))
    folder_to_process = os.path.join(base_image_folder, sub_folder)
    for image_filename in [x for x in os.listdir(folder_to_process)]:
        print('to process image {}'.format(image_filename))
        image_path = os.path.join(folder_to_process, image_filename)
        resize_img(image_path)