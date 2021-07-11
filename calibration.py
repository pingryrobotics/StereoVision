import cv2 as cv
import numpy as np
import os
import glob
from enum import Enum 



class CameraCalibration():
    """
    Class to calibrate a camera and undistort images for the camera
    (adding specific cameras needs to be added)
    """

    def __init__(self, cameraName:str, img_dir:str, cameraNum:int):
        """
        Initialize the checkerboard with a set of checkerboard dimensions

        Args:
            - cb_dims:tuple (height, width) - the dimensions of the checkerboard in terms of inner corners
                (the locations where black squares touch each other)
            - cameraName:str - the name of the camera
            - img_dir:str - the directory of the stored images
            - cameraNum:int - the number of the camera
        """
        # not actually sure what the 30 and 0.001 are, i think 0.001 is error though
        self.criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        self.cal_dir = img_dir
        self.cameraNum = cameraNum
        self.cameraName = cameraName

    def calibrate(self, cb_dims) -> None:
        """
        Calibrate the camera with stored images and stores the result in the calibration info class

        """
        # array of real world points (object points) of checkerboard dimensions from all images
        all_obj_points = []
        # array of 2d points on the image for each checkerboard
        all_img_points = []

        # array of real points for each image
        # real points are (x, y, z) except z is always 0 since you assume the checkerboard isnt moving but the camera is
        # (even tho thats not true its okay)
        object_points = np.zeros((1, cb_dims[0] *cb_dims[1], 3), np.float32)
        square_size = 9
        max_square_size = square_size * cb_dims[0]
        # this reshapes it into uhhh something
        # object_points[0,:,:2] = np.mgrid[0:self.cb_dims[0], 0:self.cb_dims[1]].T.reshape(-1, 2)
        object_points[0,:,:2] = np.mgrid[0:max_square_size:square_size, 0:max_square_size:square_size].T.reshape(-1, 2)


        # get path of all images in the image folder
        image_list = glob.glob(f'{self.cal_dir}/calibration_images/*.png')

        # get corner points on each image
        for image_path in image_list:
            # load image from path
            img = cv.imread(image_path)
            # convert to grayscale
            grayscale = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

            # find checkerboard corners
            ret, corners = cv.findChessboardCorners(image=grayscale, patternSize=cb_dims, corners=None)

            # if found, add points to total object point list
            if ret == True:
                # add object points to big boy list
                all_obj_points.append(object_points)
                
                # refine corner points
                refined_corners = cv.cornerSubPix(
                    image=grayscale,
                    corners=corners,
                    winSize=(11,11), # half the side length of the search window (ima be real, no idea why its 11)
                    zeroZone=(-1,-1), # indicates there's no zero zone
                    criteria=self.criteria)

                all_img_points.append(refined_corners)
                cv.drawChessboardCorners(img, cb_dims, refined_corners, ret)
                cv.imshow('img', img)
                cv.waitKey(500) # wait for a key to be pressed for .5 seconds
        cv.destroyAllWindows()

        # actually calibrate the camera
        # output is camera matrix, distortion coefficients, rotation & translation vectors
        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
            objectPoints=all_obj_points, # all the 3d points from the images
            imagePoints=all_img_points, # all the 2d points from the images
            imageSize=grayscale.shape[::-1], # the size of the images (use the last image just to get size)
            cameraMatrix=None, # no camera matrix since we dont have one
            distCoeffs=None) # ^

        # get new camera matrix and region of interest for camera after calibration
        # region of interest is the valid pixels essentially
        h, w = img.shape[:2]
        newcameramtx, roi = cv.getOptimalNewCameraMatrix(
            cameraMatrix=mtx,
            distCoeffs=dist,
            imageSize=(w,h),
            alpha=1,
            newImgSize=(w,h))
        # store info in calibration dict class
        self.calibrationDict = {
            self.CalibrationInfo.NEW_MATRIX : newcameramtx,
            self.CalibrationInfo.ORIG_MATRIX : mtx,
            self.CalibrationInfo.ROI : np.asarray(roi),
            self.CalibrationInfo.DIST_COEFFS : np.asarray(dist),
            self.CalibrationInfo.R_VECS : np.asarray(rvecs),
            self.CalibrationInfo.T_VECS : np.asarray(tvecs),
        }
        print(newcameramtx)


        self.saveCalibrations(self.calibrationDict)
    
    class CalibrationInfo(str, Enum):
        """
        Enum of calibration info
        """
        NEW_MATRIX = "NewMatrix"
        ORIG_MATRIX =  "OriginalMatrix"
        ROI = "RegionOfInterest"
        DIST_COEFFS = "DistortionCoefficients"
        R_VECS = "RotationVectors"
        T_VECS = "TranslationVectors"



    def saveCalibrations(self, calibrationDict):
        save_dir = f"{self.cal_dir}/calibrations"
        os.makedirs(save_dir, exist_ok=True)

        for key, value in calibrationDict.items():
            name = str(key)
            print(key)
            print(value)
            print(name)
            print(f"saving {key}")
            cal_file = f"{save_dir}/{key}.txt"
            self.writeMatrix(cal_file, value)

    
    def writeMatrix(self, filename, data):
        with open(filename, 'w') as outfile:
            outfile.write('# {0}\n'.format(data.shape))

            if data.ndim == 1:
                np.savetxt(outfile, data, fmt='%-7.0f')
                outfile.write('# New slice\n')
            
            else:
                for data_slice in data:
                    np.savetxt(outfile, data_slice, fmt='%-7.0f')
                    outfile.write('# New slice\n')
    
    def loadCalibrations(self):
        calibration_dir = f"{self.cal_dir}/calibrations"
        calibration_dict = {}
        for key in self.CalibrationInfo:
            filename = f"{calibration_dir}/{key}.txt"
            with open(filename) as file:
                shapeStr = file.readline().strip()
                shapeStr = shapeStr.split("# ", 1)[1]
                shape = eval(shapeStr)
            calibration_dict[key] = self.readMatrix(filename, shape)
            # print(calibration_dict)
        # print(calibration_dict[self.CalibrationInfo.DIST_COEFFS])
        self.calibrationDict = calibration_dict
            


    def readMatrix(self, filename, data_shape):
        data = np.loadtxt(filename) if os.path.exists(filename) else None
        data = data.reshape(data_shape)
        if "RegionOfInterest" in filename:
            data = data.astype(int)
        return data


    def takeCalibrationPhotos(self):
        cam = cv.VideoCapture(self.cameraNum)
        
        cv.namedWindow("Camera Calibration")
        img_counter = 0
        print("taking photos")

        while True:
            ret, frame = cam.read()
            if not ret:
                print("failed to grab frame")
                break
            cv.imshow("test", frame)

            k = cv.waitKey(1)
            if k%256 == 81:
                # Q pressed
                print("Q hit, closing...")
                break
            elif k%256 == 32:
                # SPACE pressed
                save_dir = f"{self.cal_dir}/images"
                os.makedirs(save_dir, exist_ok=True)
                img_name = "opencv_frame_{}.png".format(img_counter)
                cv.imwrite(f"{save_dir}/{img_name}", frame)
                print("{} written!".format(img_name))
                img_counter += 1

        cam.release()

        cv.destroyAllWindows()



    def undistort(self, img) -> np.ndarray:
        """
        Undistorts the provided image

        Args:
            - img: array_like - the image to undistort

        Returns:
            - np.ndarray (type might be wrong) - the undistorted image
        """
        if self.calibrationDict is None:
            return None
        
        undistorted = cv.undistort(
            src=img,
            cameraMatrix=self.calibrationDict[self.CalibrationInfo.ORIG_MATRIX],
            distCoeffs=self.calibrationDict[self.CalibrationInfo.DIST_COEFFS],
            dst=None,
            newCameraMatrix=self.calibrationDict[self.CalibrationInfo.NEW_MATRIX])

        # x, y, w, h = self.calibrationDict[self.CalibrationInfo.ROI]
        # cropped = undistorted[y:y+h, x:x+w]
        return undistorted


    # class CalibrationInfo():
    #     """
    #     Subclass to store calibration info
    #     """
    #     def __init__(self, newMatrix, roi, matrix, distCoeffs, rvecs, tvecs):
    #         self.newCameraMatrix = newMatrix
    #         self.roi = roi
    #         self.cameraMatrix = matrix
    #         self.distCoeffs = distCoeffs
    #         self.rvecs = rvecs
    #         self.tvecs = tvecs


def calibrateCameras(left_cam_cal, right_cam_cal):

    left_cam_cal.takeCalibrationPhotos()
    # left_cam_cal.calibrate([7,7])

    right_cam_cal.takeCalibrationPhotos()
    # right_cam_cal.calibrate([7,7])


def testCalibrations(left_cam_cal:CameraCalibration, right_cam_cal:CameraCalibration, exp_num):
    left_cam_cal.loadCalibrations()
    right_cam_cal.loadCalibrations()

    left_im = cv.imread(f".local/exp-{exp_num}/{left_cam_cal.cameraName}/calibration_images/img-0.png")
    right_im = cv.imread(f".local/exp-{exp_num}/{right_cam_cal.cameraName}/calibration_images/img-0.png")

    left_calibrated = left_cam_cal.undistort(left_im)
    right_calibrated = right_cam_cal.undistort(right_im)
    while True:
        # cv.namedWindow("Calibrated Images")
        joined = np.hstack((left_calibrated, right_calibrated))
        cv.imshow("Left Camera | Right Camera", joined)
        k = cv.waitKey(1)
        if k%256 == 81:
        # shift + Q pressed
            break

exp_num = 2
cam_path = f".local/exp-{exp_num}"

left_cam_id = 2
left_cam_name = f"left"
left_cam_cal = CameraCalibration(left_cam_name, f"{cam_path}/{left_cam_name}", left_cam_id)

right_cam_id = 1
right_cam_name = f"right"
right_cam_cal = CameraCalibration(right_cam_name, f"{cam_path}/{right_cam_name}", right_cam_id)

# testCalibrations(left_cam_cal, right_cam_cal, exp_num)
# calibrateCameras(left_cam_cal, right_cam_cal)



# cameracal1.loadCalibrations()

# new_im = cv.imread(f".local/{camera_name}/opencv_frame_0.png")
# cropped = cameracal1.undistort(new_im)
# path = f".local/{camera_name}/calibrated"
# os.makedirs(path, exist_ok=True)
# cv.imwrite(f"{path}/cropped.png", cropped)