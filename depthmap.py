from calibration import CameraCalibration
import os
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from pathlib import Path



def createDepthMap(left_img, right_img):
    stereo = cv.StereoBM_create(numDisparities = 16, blockSize = 17)
    depth = stereo.compute(right_img, left_img)
    both_im = np.hstack((left_img, right_img))
    cv.imshow("Left", both_im)
    plt.imshow(depth)
    plt.axis("off")
    plt.show()


def takeStereoPhotos(save_dir):
        cam_left = cv.VideoCapture(2)
        cam_right = cv.VideoCapture(1)
        
        cv.namedWindow("Camera Calibration")
        img_counter = 0
        print("taking photos")

        while True:
            retL, frameL = cam_left.read()
            retR, frameR = cam_right.read()
            if not retL and not retR:
                print("failed to grab frame")
                break
            both_frames = np.hstack((frameL, frameR))
            cv.imshow("Left Camera | Right Camera", both_frames)

            k = cv.waitKey(1)
            if k%256 == 81:
                # Q pressed
                print("Q hit, closing...")
                break
            elif k%256 == 32:
                # SPACE pressed
                left_save_dir = f"{save_dir}/left/calibration_images"
                right_save_dir = f"{save_dir}/right/calibration_images"
                Path(left_save_dir).mkdir(exist_ok=True, parents=True)
                Path(right_save_dir).mkdir(exist_ok=True, parents=True)
                img_name = "img-{}.png".format(img_counter)
                cv.imwrite(f"{left_save_dir}/{img_name}", frameL)
                cv.imwrite(f"{right_save_dir}/{img_name}", frameR)
                print("L and R image {} written".format(img_counter))
                img_counter += 1

        cam_left.release()
        cam_right.release()

        cv.destroyAllWindows()

save_dir = ".local/exp-2"
takeStereoPhotos(save_dir)


# left_cal = CameraCalibration("left_camera", ".local/left_camera", 2)
# right_cal = CameraCalibration("right_camera", ".local/right_camera", 1)

# left_cal.loadCalibrations()
# right_cal.loadCalibrations()

# left_im_path = ".local/stereo/images/left/img-0.png"
# right_im_path = ".local/stereo/images/right/img-0.png"
# left_im_raw = cv.imread(left_im_path, 0)
# right_im_raw = cv.imread(right_im_path, 0)

# left_undist = left_cal.undistort(left_im_raw)
# right_undist = right_cal.undistort(right_im_raw)

# # left_undist = cv.imread("/Users/oliviataylor/Downloads/left.png", 0)
# # right_undist = cv.imread("/Users/oliviataylor/Downloads/right.png", 0)

# createDepthMap(left_undist, right_undist)