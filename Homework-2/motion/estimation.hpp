/*
�����Ҫ�Ӵ���OpenCV��������Ҫ��ģ�飺
��1��features2dģ�飺����������ȡ��ƥ��Ҫ�õ���ģ��
��2��calib3dģ�飺����OpenCVʵ�ֵ�һ�����У׼����̬����ģ��
*/

#ifndef ESTIMATION_H
#define ESTIMATION_H

#include "feature_extract_match.hpp"

//opencv
#include <opencv2/calib3d/calib3d.hpp>

//����������˶�������R��t
void pose_estimation_2d2d(vector<KeyPoint> key_points_1, vector<KeyPoint> key_points_2, vector<DMatch> matches, Mat& R, Mat& t)
{
    //�ڲξ���
    double fx, fy, cx, cy;
    Mat K = Mat::eye(3, 3, CV_64FC1);   //Mat::eye()����һ��ָ����С�����͵ĺ㶨����
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

    //�����е�KeyPointת��ΪPoint2f
    vector<Point2f> points_1;
    vector<Point2f> points_2;
    for (int i = 0; i < matches.size(); i++)
    {
        points_1.push_back(key_points_1[matches[i].queryIdx].pt);   //pt������KeyPoint������
        points_2.push_back(key_points_2[matches[i].trainIdx].pt);
    }
    //�����������
    Mat fundamental_matrix;
    fundamental_matrix = findFundamentalMat(points_1, points_2, FM_8POINT);  //���һ������CV_FM_8POINT��ʾʹ��8�㷨�����������
    cout << "fundamental_matrix = " << endl;
    cout << fundamental_matrix << endl;

    //ֱ�ӵ���findEssentialMat()���������ʾ���findEssentialMat()������ԭ�ͣ�����Ҫ���õ�Ϊ������һ��
    //findEssentialMat(InputArray points1, InputArray points2, double focal = 1.0, Point2d pp = Point2d(0, 0), int method = RANSAC, double prob = 0.999, double threshold = 1.0, OutputArray mask = noArray());
    double focal = 521;
    Point2d pp(325.1, 249.7);
    Mat essential_matrix;
    essential_matrix = findEssentialMat(points_1, points_2, focal, pp);
    cout << "essential_matrix = " << endl;
    cout << essential_matrix << endl;

    //�ӱ��ʾ���ָ�R��t
    //recoverPose(InputArray E, InputArray points1, InputArray points2, OutputArray R, OutputArray t, double focal = 1.0, Point2d pp = Point2d(0, 0), InputOutputArray = noArray());
    recoverPose(essential_matrix, points_1, points_2, R, t, focal, pp);
}

//���㵥Ӧ����
void homograph_estimation(vector<KeyPoint> key_points_1, vector<KeyPoint> key_points_2, vector<DMatch> matches, Mat& H)
{
    //�����е�KeyPointת��ΪPoint2f
    vector<Point2f> points_1, points_2;
    for (int i = 0; i < matches.size(); i++)
    {
        points_1.push_back(key_points_1[matches[i].queryIdx].pt);
        points_2.push_back(key_points_2[matches[i].trainIdx].pt);
    }

    //���㵥Ӧ����
    H = findHomography(points_1, points_2, RANSAC);
    cout << "H = " << endl;
    cout << H << endl;
}

#endif