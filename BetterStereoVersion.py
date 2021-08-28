import cv2 as cv
import numpy as np
import os
import glob
import time
from enum import Enum 
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path



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
        self.StereoMapX = []
        self.StereoMapY = []

    def calibrate(cb_dims) -> None:
        """
        Calibrate the camera with stored images and stores the result in the calibration info class
        """
        # array of real world points (object points) of checkerboard dimensions from all images
        all_obj_points = []
        all_obj_pointsL = []
        all_obj_pointsR = []
        # array of 2d points on the image for each checkerboard
        img_ptsL1 = []
        img_ptsR1 = []
        img_ptsL2 = []
        img_ptsR2 = []

        # array of real points for each image
        # real points are (x, y, z) except z is always 0 since you assume the checkerboard isnt moving but the camera is
        # (even tho thats not true its okay)
        object_points = np.zeros((cb_dims[0] *cb_dims[1], 3), np.float32)
        square_size = 30
        max_square_size = square_size * cb_dims[0]
        # this reshapes it into uhhh something
        # object_points[0,:,:2] = np.mgrid[0:self.cb_dims[0], 0:self.cb_dims[1]].T.reshape(-1, 2)
        object_points[:,:2] = np.mgrid[0:7, 0:7].T.reshape(-1, 2)


        # get path of all images in the image folder
        image_listR = sorted(glob.glob(f'{right_cam_cal.cal_dir}/images/*.png'))
        image_listL = sorted(glob.glob(f'{left_cam_cal.cal_dir}/images/*.png'))
        # get corner points on each image
        for i in range(100):
            # load image from paths
            imgL = cv.imread(image_listL[i])
            imgR = cv.imread(image_listR[i])
            # convert to grayscale
            grayscaleL = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
            grayscaleR = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)
            # find checkerboard corners
            retL, cornersL = cv.findChessboardCorners(image=grayscaleL, patternSize=cb_dims, corners=None)
            retR, cornersR = cv.findChessboardCorners(image=grayscaleR, patternSize=cb_dims, corners=None)

            # if found, add points to total object point list
            if retL & retR:
                # add object points to big boy list
                all_obj_points.append(object_points)
                all_obj_pointsL.append(object_points)
                all_obj_pointsR.append(object_points)
                
                refined_cornersL = cv.cornerSubPix(
                    image=grayscaleL,
                    corners=cornersL,
                    winSize=(11,11),
                    zeroZone=(-1,-1),
                    criteria=left_cam_cal.criteria)

                refined_cornersR = cv.cornerSubPix(
                    image=grayscaleR,
                    corners=cornersR,
                    winSize=(11,11),
                    zeroZone=(-1,-1), 
                    criteria=right_cam_cal.criteria)
                

                img_ptsL1.append(refined_cornersL)
                img_ptsR1.append(refined_cornersR)
                img_ptsL2.append(refined_cornersL)
                img_ptsR2.append(refined_cornersR)
                #cv.drawChessboardCorners(imgL, cb_dims, refined_cornersL, retL)
                #cv.drawChessboardCorners(imgR, cb_dims, refined_cornersR, retR)
                #cv.imshow('imgL', imgL)
                #cv.imshow('imgR', imgR)
                #cv.waitKey(500) # wait for a key to be pressed for .5 seconds
            elif retL & (retR == False):
                all_obj_pointsL.append(object_points)
                refined_cornersL = cv.cornerSubPix(
                    image=grayscaleL,
                    corners=cornersL,
                    winSize=(11,11),
                    zeroZone=(-1,-1),
                    criteria=left_cam_cal.criteria)

                img_ptsL1.append(refined_cornersL)

            elif retR & (retL == False):
                all_obj_pointsR.append(object_points)
                # refine corner points
                refined_cornersR = cv.cornerSubPix(
                    image=grayscaleR,
                    corners=cornersR,
                    winSize=(11,11),
                    zeroZone=(-1,-1), 
                    criteria=right_cam_cal.criteria)

                img_ptsR1.append(refined_cornersL)
                


        cv.destroyAllWindows()


        retL, mtxL, distL, rvecsL, tvecsL = cv.calibrateCamera(
            objectPoints=all_obj_pointsL,
            imagePoints=img_ptsL1, 
            imageSize=grayscaleL.shape[::-1], 
            cameraMatrix=None, 
            distCoeffs=None) 

        retR, mtxR, distR, rvecsR, tvecsR = cv.calibrateCamera(
            objectPoints=all_obj_pointsR,
            imagePoints=img_ptsR1,
            imageSize=grayscaleR.shape[::-1],
            cameraMatrix=None,
            distCoeffs=None) 
        hL, wL = imgL.shape[:2]
        newcameramtxL, roiL = cv.getOptimalNewCameraMatrix(
            cameraMatrix=mtxL,
            distCoeffs=distL,
            imageSize=(wL,hL),
            alpha=1,
            newImgSize=(wL,hL))

        hR, wR = imgR.shape[:2]
        newcameramtxR, roiR = cv.getOptimalNewCameraMatrix(
            cameraMatrix=mtxR,
            distCoeffs=distR,
            imageSize=(wR,hR),
            alpha=1,
            newImgSize=(wR,hR))
        # store info in calibration dict class
        left_cam_cal.calibrationDict = {
            left_cam_cal.CalibrationInfo.NEW_MATRIX : newcameramtxL,
            left_cam_cal.CalibrationInfo.ORIG_MATRIX : mtxL,
            left_cam_cal.CalibrationInfo.ROI : np.asarray(roiL),
            left_cam_cal.CalibrationInfo.DIST_COEFFS : np.asarray(distL),
            left_cam_cal.CalibrationInfo.R_VECS : np.asarray(rvecsL),
            left_cam_cal.CalibrationInfo.T_VECS : np.asarray(tvecsL),
        }

        right_cam_cal.calibrationDict = {
            right_cam_cal.CalibrationInfo.NEW_MATRIX : newcameramtxR,
            right_cam_cal.CalibrationInfo.ORIG_MATRIX : mtxR,
            right_cam_cal.CalibrationInfo.ROI : np.asarray(roiR),
            right_cam_cal.CalibrationInfo.DIST_COEFFS : np.asarray(distR),
            right_cam_cal.CalibrationInfo.R_VECS : np.asarray(rvecsR),
            right_cam_cal.CalibrationInfo.T_VECS : np.asarray(tvecsR),
        }
        print(newcameramtxL)
        frameLUndistorted = left_cam_cal.undistort(imgL)
        frameRUndistorted = right_cam_cal.undistort(imgR)
        cv.imshow("Left", frameLUndistorted)
        cv.imshow("Right", frameRUndistorted)
        cv.waitKey(0)
        #flags = 0
        flags = cv.CALIB_FIX_INTRINSIC
        #flags = cv.CALIB_FIX_PRINCIPAL_POINT
        #flags |= cv.CALIB_USE_INTRINSIC_GUESS
        #flags|= cv.CALIB_USE_EXTRINSIC_GUESS
        #flags = cv.CALIB_FIX_ASPECT_RATIO
        #flags |= cv.CALIB_FIX_FOCAL_LENGTH
        #flags |= cv.CALIB_FIX_ASPECT_RATIO
        #flags |= cv.CALIB_ZERO_TANGENT_DIST
        #flags |= cv.CALIB_RATIONAL_MODEL
        flags |= cv.CALIB_SAME_FOCAL_LENGTH
        #flags = cv.CALIB_FIX_K3
        #flags |= cv.CALIB_FIX_K4
        #flags |= cv.CALIB_FIX_K5

        criteria_stereo= (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        

        # imgL = imgL[10:770, 10:450]
        # imgR = imgR[10:770, 10:450]
        
        # cv.waitKey(0)

        rectify_scale= 0
        stereocalibration_criteria = (cv.TERM_CRITERIA_MAX_ITER + cv.TERM_CRITERIA_EPS, 100, 1e-5)
        rms, leftMatrix, leftDistortion, rightMatrix, rightDistortion, rotationMatrix, translationVector, E, F =  cv.stereoCalibrate(
        all_obj_points, img_ptsL2, img_ptsR2,
        newcameramtxL, distL,
        newcameramtxR, distR,
        grayscaleL.shape[::-1], criteria = stereocalibration_criteria,
        flags = flags)
        rect_l, rect_r, proj_mat_l, proj_mat_r, Q, roiL, roiR= cv.stereoRectify(leftMatrix, leftDistortion, rightMatrix, rightDistortion, grayscaleL.shape[::-1], rotationMatrix, translationVector, rectify_scale, 0, alpha = -1)
        
        Left_Stereo_MapX, Left_Stereo_MapY= cv.initUndistortRectifyMap(leftMatrix, leftDistortion, rect_l, proj_mat_l,
                                                    grayscaleL.shape[::-1], cv.CV_32FC1)
        Right_Stereo_MapX, Right_Stereo_MapY= cv.initUndistortRectifyMap(leftMatrix, leftDistortion, rect_r, proj_mat_r,
                                                    grayscaleR.shape[::-1], cv.CV_32FC1)
        left_cam_cal.StereoMapX = Left_Stereo_MapX
        left_cam_cal.StereoMapY = Left_Stereo_MapY
        right_cam_cal.StereoMapX = Right_Stereo_MapX
        right_cam_cal.StereoMapY = Right_Stereo_MapY
        left_cam_cal.saveCalibrations(left_cam_cal.calibrationDict, np.asarray(left_cam_cal.StereoMapX), np.asarray(left_cam_cal.StereoMapY))
        right_cam_cal.saveCalibrations(right_cam_cal.calibrationDict, np.asarray(right_cam_cal.StereoMapX), np.asarray(right_cam_cal.StereoMapY))
        Left_nice= cv.remap(grayscaleL,Left_Stereo_MapX,Left_Stereo_MapY, cv.INTER_LANCZOS4, cv.BORDER_CONSTANT, 0)  # Rectify the image using the kalibration parameters founds during the initialisation
        Right_nice= cv.remap(grayscaleR,Right_Stereo_MapX,Right_Stereo_MapY, cv.INTER_LANCZOS4, cv.BORDER_CONSTANT, 0)
        
        stereo = cv.StereoBM_create(numDisparities = 16, blockSize = 21)
        # stereo.setMinDisparity(4)
        # stereo.setNumDisparities(128)
        # stereo.setBlockSize(21)
        # stereo.setSpeckleRange(16)
        # stereo.setSpeckleWindowSize(45)

        depth = stereo.compute(Left_nice, Right_nice)
        both_im = np.hstack((Left_nice, Right_nice))
        cv.imshow("Left", both_im)
        plt.imshow(depth)
        plt.axis("off")
        plt.show()

        print("Saving paraeters ......")
        cv.imshow("Left image before rectification", imgL)
        cv.imshow("Right image before rectification", imgR)

        #Left_nice= cv.remap(imgL,Left_Stereo_Map[0],Left_Stereo_Map[1], cv.INTER_LANCZOS4, cv.BORDER_CONSTANT, 0)
        #Right_nice= cv.remap(imgR,Right_Stereo_Map[0],Right_Stereo_Map[1], cv.INTER_LANCZOS4, cv.BORDER_CONSTANT, 0)

        # cv.imshow("Left image after rectification", Left_nice)
        # cv.imshow("Right image after rectification", Right_nice)
        # cv.waitKey(0)

        # cv.imshow("Output image", out)
        # cv.waitKey(0)


        

    
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



    def saveCalibrations(self, calibrationDict, Stereo_MapX, Stereo_MapY):
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
        
        self.writeMatrix(f"{save_dir}/StereoMapX.txt", Stereo_MapX)
        self.writeMatrix(f"{save_dir}/StereoMapY.txt", Stereo_MapY)

    
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
        filename = f"{calibration_dir}/StereoMapX.txt"
        with open(filename) as file:
                shapeStr = file.readline().strip()
                shapeStr = shapeStr.split("# ", 1)[1]
                shape = eval(shapeStr)
        self.StereoMapX = self.readMatrix(filename, shape)

        filename = f"{calibration_dir}/StereoMapY.txt"
        with open(filename) as file:
                shapeStr = file.readline().strip()
                shapeStr = shapeStr.split("# ", 1)[1]
                shape = eval(shapeStr)
        self.StereoMapY = self.readMatrix(filename, shape)
            
    def showLiveDisparity():
        window_size = 3
        min_disp = 2
        num_disp = 130-min_disp
        #left_cam_cal.loadCalibrations()
        #right_cam_cal.loadCalibrations()

        stereo = cv.StereoSGBM_create(minDisparity = min_disp,
            numDisparities = num_disp,
            blockSize = window_size,
            uniquenessRatio = 10,
            speckleWindowSize = 100,
            speckleRange = 32,
            disp12MaxDiff = 5,
            P1 = 8*3*window_size**2,
            P2 = 32*3*window_size**2)

        stereoR=cv.StereoSGBM_create(minDisparity = min_disp,
            numDisparities = num_disp,
            blockSize = window_size,
            uniquenessRatio = 10,
            speckleWindowSize = 100,
            speckleRange = 32,
            disp12MaxDiff = 5,
            P1 = 8*3*window_size**2,
            P2 = 32*3*window_size**2)# Crea
        camLeft = cv.VideoCapture(1)
        camRight = cv.VideoCapture(0)
        while True:
            retL, frameL= camLeft.read()
            retR, frameR= camRight.read()
            frameL = frameL[20:780, 20:460]
            frameR = frameR[20:780, 20:460]
            grayscaleL = cv.cvtColor(frameL, cv.COLOR_BGR2GRAY)
            grayscaleR = cv.cvtColor(frameR, cv.COLOR_BGR2GRAY)
            Left_nice= cv.remap(grayscaleL,left_cam_cal.StereoMapX,left_cam_cal.StereoMapY, cv.INTER_LANCZOS4, cv.BORDER_CONSTANT, 0)  # Rectify the image using the kalibration parameters founds during the initialisation
            Right_nice= cv.remap(grayscaleR,right_cam_cal.StereoMapX,right_cam_cal.StereoMapY, cv.INTER_LANCZOS4, cv.BORDER_CONSTANT, 0)
            cv.imshow('Both Images', np.hstack([Left_nice, Right_nice]))
            cv.imshow('Normal', np.hstack([frameL, frameR]))

            #grayR= cv.cvtColor(Right_nice,cv.COLOR_BGR2GRAY)
            #grayL= cv.cvtColor(Left_nice,cv.COLOR_BGR2GRAY)

            disp= stereo.compute(Left_nice,Right_nice)
            dispL= disp
            dispR= stereoR.compute(Right_nice,Left_nice)
            dispL= np.int16(dispL)
            dispR= np.int16(dispR)

            disp= ((disp.astype(np.float32)/ 16)-min_disp)/num_disp

            kernel= np.ones((3,3),np.uint8)
            closing= cv.morphologyEx(disp,cv.MORPH_CLOSE, kernel)

            dispc= (closing-closing.min())*255
            dispC= dispc.astype(np.uint8)                                   
            disp_Color= cv.applyColorMap(dispC,cv.COLORMAP_OCEAN)    
            disp_ColorL= cv.applyColorMap(dispL,cv.COLORMAP_OCEAN)    
            disp_ColorR= cv.applyColorMap(dispR,cv.COLORMAP_OCEAN)         
            cv.imshow('Filtered Color Depth', disp_Color)
            #cv.imshow('Filtered Color DepthL', disp_ColorL)
            #cv.imshow('Filtered Color DepthR', disp_ColorR)
            if cv.waitKey(1) & 0xFF == ord(' '):
                break
        cv.destroyAllWindows()

    def readMatrix(self, filename, data_shape):
        data = np.loadtxt(filename) if os.path.exists(filename) else None
        data = data.reshape(data_shape)
        if "RegionOfInterest" in filename:
            data = data.astype(int)
        return data


    def takeCalibrationPhotos():
        camLeft = cv.VideoCapture(1)
        camRight = cv.VideoCapture(0)
        
        
        cv.namedWindow("Camera Calibration")
        img_counter = 0
        print("taking photos")

        while True:
            retL, frameL = camLeft.read()
            retR, frameR = camRight.read()
            frameL = frameL[20:780, 20:460]
            frameR = frameR[20:780, 20:460]
            # print("Go")
            # frameR = left_cam_cal.undistort(frameR)
            grayL = cv.cvtColor(frameL, cv.COLOR_BGR2GRAY)
            grayR = cv.cvtColor(frameR, cv.COLOR_BGR2GRAY)
            if not retL and not retR:
                print("failed to grab frame")
                break
            both_frames = np.hstack((frameL, frameR))
            cv.imshow("Left Camera | Right Camera", both_frames)
            k = cv.waitKey(1)
            time.sleep(0.5)
            if k%256 == 81:
                # Q pressed
                print("Q hit, closing...")
                break
            elif img_counter < 100:
                # SPACE pressed
                left_save_dir = f".local/exp-2/left/images"
                right_save_dir = f".local/exp-2/right/images"
                Path(left_save_dir).mkdir(exist_ok=True, parents=True)
                Path(right_save_dir).mkdir(exist_ok=True, parents=True)
                img_name = "img-{}.png".format(img_counter)
                cv.imwrite(f"{left_save_dir}/{img_name}", frameL)
                cv.imwrite(f"{right_save_dir}/{img_name}", frameR)
                print("L and R image {} written".format(img_counter))
                img_counter += 1
            else:
                break
        camLeft.release()
        camRight.release()

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

    # left_im = cv.imread(f".local/exp-{exp_num}/{left_cam_cal.cameraName}/calibration_images/img-0.png")
    # right_im = cv.imread(f".local/exp-{exp_num}/{right_cam_cal.cameraName}/calibration_images/img-0.png")
    camLeft = cv.VideoCapture(left_cam_cal.cameraNum, cv.CAP_DSHOW)
    camRight = cv.VideoCapture(right_cam_cal.cameraNum, cv.CAP_DSHOW)    
    camLeft.set(3, 780)
    camRight.set(3, 780)

    retL, frameL = camLeft.read()
    retR, frameR = camRight.read()
    
    left_calibrated = left_cam_cal.undistort(frameL)
    right_calibrated = right_cam_cal.undistort(frameR)
    while True:
        # cv.namedWindow("Calibrated Images")
        joinedOriginal = np.hstack((frameL, frameR))
        joinedNew = np.hstack((left_calibrated, right_calibrated))
        cv.imshow("Original Left Camera | Right Camera", joinedOriginal)
        cv.imshow("Left Camera | Right Camera", joinedNew)
        k = cv.waitKey(1)
        if k%256 == 81:
        # shift + Q pressed
            break

exp_num = 2
cam_path = f".local/exp-{exp_num}"

left_cam_id = 1
left_cam_name = f"left"
left_cam_cal = CameraCalibration(left_cam_name, f"{cam_path}/{left_cam_name}", left_cam_id)

right_cam_id = 0
right_cam_name = f"right"
right_cam_cal = CameraCalibration(right_cam_name, f"{cam_path}/{right_cam_name}", right_cam_id)
#CameraCalibration.takeCalibrationPhotos()#

CameraCalibration.calibrate((7,7))
CameraCalibration.showLiveDisparity()
