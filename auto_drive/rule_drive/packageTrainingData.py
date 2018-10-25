# based on Chad's code
# https://adc.github.trendmicro.com/chad-chen/pumpkin/blob/master/scripts/collect_training_data.py
# run under the folder with IMG: xxxx\formula-trend-1.0.0-beta.3\formula-trend-1.0.0-beta.3\Windows\Log
# usage: python packageTrainingData.py teamId trackNum setCount
# example: python .\packageTrainingData.py 309 3 6
# Important: remember to clear the folder "xxxx\formula-trend-1.0.0-beta.3\Windows\Log" before you start the simulator recording
import os
import shutil
import subprocess
import sys
import tarfile
from re import match
from time import gmtime, strftime


DRIVING_LOG = "driving_log.csv"
IMG_FOLDER = "IMG"


LEAST_IMAGE_COUNT_PER_SET = 4000
MAX_IMAGE_COUNT_PER_SET = 12000
current_path = os.path.abspath(os.path.dirname(sys.argv[0]))
img_folder_path = os.path.join(current_path, IMG_FOLDER)
driving_log_file_path = os.path.join(current_path, DRIVING_LOG)


def generator_file_name(team_id, track_num, image_count, cost_milliseconds):
    daystr = strftime("%Y%m%d", gmtime())
    return '{}_{}_t{}_{}_{}.tar.gz'.format(daystr, team_id, track_num, image_count, cost_milliseconds)


def validate_file(file_name):
    try:
        matched = match(
            '(2018(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01]))_[0-9]+_([t][1-9]{1,2}|[r][1-6])(_[0-9]+){2}(|_[0-9a-zA-Z]*)\.tar\.gz$', file_name)
        if matched is None:
            print('File name is not matched.')
            return -1
        tar_file_path = os.path.join(current_path, file_name)
        file_size = os.stat(tar_file_path).st_size
        if file_size < 10485760 or file_size > 52428800:
            print('File size is not allowed: %s (%s)' % (file_name, file_size))
            return -2
        print('File is valid.')
        return 0
    except Exception as e:
        print("\nvalidate file failure, reason=%s" % e.strerror)
        return -1


def get_images_path():

    if not os.path.exists(img_folder_path):
        print("count_images failure due img folder does not exists")
        return 0

    if not os.path.isdir(img_folder_path):
        print("count_images failure due img folder is not directory")
        return 0

    images_path = [os.path.join(img_folder_path, name) for name in os.listdir(
        img_folder_path) if os.path.isfile(os.path.join(img_folder_path, name))]
    return images_path


def create_tar_gz_file(log_folder_path, file_name):
    try:
        print("create tar.gz start...")
        tar_file_path = os.path.join(current_path, file_name)
        tar = tarfile.open(tar_file_path, "w:gz")

        os.chdir(log_folder_path)
        tar.add(DRIVING_LOG)
        tar.add(IMG_FOLDER, recursive=True)
        tar.close()        
        print("create tar.gz done, tar_file_path=%s" % tar_file_path)
        return 0
    except Exception as e:
        print("create tar.gz failure, reason=%s" % e.strerror)
        return -1


def main():
    if len(sys.argv)!=4:
        print('usage: python packageTrainingData.py teamId trackNum setCount')
        print('example: python packageTrainingData.py 309 4 10')
        return
    
    teamId = int(sys.argv[1])
    trackNum = int(sys.argv[2])
    setCount = int(sys.argv[3])

    print("current_path=%s" % current_path)        
    images_path = get_images_path()
    image_count = len(images_path)

    driving_log_lines = []
    with open(driving_log_file_path) as f:
        driving_log_lines = f.readlines()

        
    print("image_count=%d" % image_count)
    if image_count <= 0:
        return -1

    image_count_per_set = int((image_count-250)/setCount)
    if image_count_per_set<LEAST_IMAGE_COUNT_PER_SET or image_count_per_set>MAX_IMAGE_COUNT_PER_SET:
        print('image_count_per_set is %d should between %d and %d' % (image_count_per_set, LEAST_IMAGE_COUNT_PER_SET, MAX_IMAGE_COUNT_PER_SET))
        return

    # move these images to some sub folders and also the drive log, and then do package
    image_moved_index = 0
    for i in range(setCount):
        sub_driving_log_lines = []
        sub_folder = os.path.join(current_path, str(i))
        if not os.path.exists(sub_folder):
            os.mkdir(sub_folder)
        sub_folder_img =  os.path.join(sub_folder, 'IMG')
        if not os.path.exists(sub_folder_img):
            os.mkdir(sub_folder_img)
        for j in range(image_count_per_set+i*5):            
            # move files to sub_folder
            image_path = images_path[image_moved_index]
            folder,image_filename = os.path.split(image_path)
            driving_log_line = driving_log_lines[image_moved_index]
            sub_driving_log_lines.append(driving_log_line)
            shutil.move(image_path, os.path.join(sub_folder_img, image_filename))
            image_moved_index += 1
        driving_log_to_write = os.path.join(sub_folder, 'driving_log.csv')
        with open(driving_log_to_write, 'w') as f:
            f.writelines(sub_driving_log_lines)

        image_count = image_count_per_set+i*5
        cost_milliseconds = int(image_count*1000.0/15)
        tar_file_name = generator_file_name(teamId, trackNum, image_count, cost_milliseconds)
        create_tar_gz_file(sub_folder, tar_file_name)        
        os.chdir(current_path)
        validate_file(tar_file_name)


if __name__ == "__main__":
    sys.exit(main())
