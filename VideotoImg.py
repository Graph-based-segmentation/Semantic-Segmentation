import cv2
import os
import paths

def read_frame(idx):

    while(True):
        num_frame = 10
        idx += 1

        for num in range(0, num_frame):
            _, img = cam.read()

        name = '{:06d}'.format(idx) + '.png'
        print('Creating...', name)

        cv2.imwrite(os.path.join(paths.video_path + '/Hanyang_20190108_2', name), img)

    cam.release()
    cv2.destroyAllWindows()

def main(idx, current_frame):
    """reade frame_by_frame video"""
    while (True):
        boolean, frame = cam.read()
        idx += 1
        if boolean:
            name = '{:06d}'.format(idx) + '.png'
            print('Creating...', name)

            cv2.imwrite(os.path.join(paths.video_path + '/Hanyang_20190108_2', name), frame)

            current_frame += 1
        else:
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    cam = cv2.VideoCapture(paths.test_video_path)
    current_frame = 0
    idx = 0

    try:
        if not os.path.exists(paths.video_path + '/Hanyang_20190108_2'):
            os.makedirs(paths.video_path + '/Hanyang_20190108_2')

    except OSError:
        print('Error: Creating directory of data')

    # main(idx, current_frame)
    read_frame(idx)