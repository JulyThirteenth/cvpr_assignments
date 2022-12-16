/*
在这里，要接触到OpenCV的两个重要的模块：
（1）features2d模块：进行特征提取和匹配要用到的模块
（2）calib3d模块：它是OpenCV实现的一个相机校准和姿态估计模块
*/

#ifndef ESTIMATION_H
#define ESTIMATION_H

#include "feature_extract_match.hpp"

//opencv
#include <opencv2/calib3d/calib3d.hpp>

//估计相机的运动：返回R、t
void pose_estimation_2d2d(vector<KeyPoint> key_points_1, vector<KeyPoint> key_points_2, vector<DMatch> matches, Mat& R, Mat& t)
{
    //内参矩阵
    double fx, fy, cx, cy;
    Mat K = Mat::eye(3, 3, CV_64FC1);   //Mat::eye()返回一个指定大小和类型的恒定矩阵
    fx = 836.9;
    fy = 627.3;
    cx = 980.1;
    cy = 556.8;
    K.at<double>(0, 0) = fx;
    K.at<double>(0, 2) = cx;
    K.at<double>(1, 1) = fy;
    K.at<double>(1, 2) = cy;
    cout << "internal reference matrix = " << endl;
    cout << K << endl;

    //将所有的KeyPoint转化为Point2f
    vector<Point2f> points_1;
    vector<Point2f> points_2;
    for (int i = 0; i < matches.size(); i++)
    {
        points_1.push_back(key_points_1[matches[i].queryIdx].pt);   //pt属性是KeyPoint的坐标
        points_2.push_back(key_points_2[matches[i].trainIdx].pt);
    }
    //计算基础矩阵
    Mat fundamental_matrix;
    fundamental_matrix = findFundamentalMat(points_1, points_2, FM_8POINT);  //最后一个参数CV_FM_8POINT表示使用8点法计算基础矩阵
    cout << "fundamental_matrix = " << endl;
    cout << fundamental_matrix << endl;

    //直接调用findEssentialMat()函数来求本质矩阵，findEssentialMat()有两种原型，本次要调用的为下面这一种
    //findEssentialMat(InputArray points1, InputArray points2, double focal = 1.0, Point2d pp = Point2d(0, 0), int method = RANSAC, double prob = 0.999, double threshold = 1.0, OutputArray mask = noArray());
    double focal = 521;
    Point2d pp(325.1, 249.7);
    Mat essential_matrix;
    essential_matrix = findEssentialMat(points_1, points_2, focal, pp);
    cout << "essential_matrix = " << endl;
    cout << essential_matrix << endl;

    //从本质矩阵恢复R、t
    //recoverPose(InputArray E, InputArray points1, InputArray points2, OutputArray R, OutputArray t, double focal = 1.0, Point2d pp = Point2d(0, 0), InputOutputArray = noArray());
    recoverPose(essential_matrix, points_1, points_2, R, t, focal, pp);
}

//计算单应矩阵
void homograph_estimation(vector<KeyPoint> key_points_1, vector<KeyPoint> key_points_2, vector<DMatch> matches, Mat& H)
{
    //将所有的KeyPoint转换为Point2f
    vector<Point2f> points_1, points_2;
    for (int i = 0; i < matches.size(); i++)
    {
        points_1.push_back(key_points_1[matches[i].queryIdx].pt);
        points_2.push_back(key_points_2[matches[i].trainIdx].pt);
    }

    //计算单应矩阵
    H = findHomography(points_1, points_2, RANSAC);
    cout << "H = " << endl;
    cout << H << endl;
}

#endif