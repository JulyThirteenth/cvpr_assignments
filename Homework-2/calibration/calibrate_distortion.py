# -*- coding: utf-8 -*-

import os

from calibrate_helper import Calibrator


def main():
    img_dir = "calibration/pictures/4.5cm"
    shape_inner_corner = (11, 8)
    size_grid = 0.045
    # create calibrator
    calibrator = Calibrator(img_dir, shape_inner_corner, size_grid)
    # calibrate the camera
    mat_intri, coff_dis = calibrator.calibrate_camera()
    # dedistort and save the dedistortion result
    save_dir = "calibration/pictures/4.5cm/dedistortion"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    calibrator.dedistortion(save_dir)


if __name__ == '__main__':
    main()