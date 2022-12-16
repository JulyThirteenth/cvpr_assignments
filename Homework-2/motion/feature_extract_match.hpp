#ifndef FEATURE_EXTRACT_MATCH_H
#define FEATURE_EXTRACT_MATCH_H

#include <iostream>
using namespace std;

//opencv
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace cv;

/*
ʵ������ͼ���������ȡ��ƥ�䣺
ʹ�����õ�Ŀ����Ϊ��ֱ�ӽ���ȡ��ƥ��Ľ��������key_point_1��key_point_2��matches��
*/
Mat feature_extract_match(Mat img1, Mat img2, vector<KeyPoint>& key_point_1, vector<KeyPoint>& key_point_2, vector<DMatch>& matches)
{

    //����ORB��ȡ������������ȡ����������ƥ����
    Ptr<FeatureDetector> orb_detectore = ORB::create();
    Ptr<DescriptorExtractor> orb_descriptor = ORB::create();
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");

    //�������
    Mat descriptor_1, descriptor_2;
    vector<DMatch> matches_before_filter;
    Mat img_matches_before_filter, img_good_matches;

    //��ȡORB�ؼ���
    orb_detectore->detect(img1, key_point_1);
    orb_detectore->detect(img2, key_point_2);

    //����������
    orb_descriptor->compute(img1, key_point_1, descriptor_1);
    orb_descriptor->compute(img2, key_point_2, descriptor_2);

    //������ƥ�䣬ƥ�����л����һЩ��ƥ��ĵ�
    matcher->match(descriptor_1, descriptor_2, matches_before_filter);

    //�޳���ƥ��ĵ㣺�����������������С����ĵ��޳�
    double min_distance = 9999;
    for (int i = 0; i < descriptor_1.rows; i++)
    {
        if (matches_before_filter[i].distance < min_distance)
        {
            min_distance = matches_before_filter[i].distance;
        }
    }
    cout << "min_distance = " << min_distance << endl;

    //�е�ʱ����С����ǳ�С����������ȡ��С����Ϊһ������ֵ30
    for (int i = 0; i < descriptor_1.rows; i++)
    {
        if (matches_before_filter[i].distance <= max(2 * min_distance, 30.0))
        {
            matches.push_back(matches_before_filter[i]);
        }
    }

    drawMatches(img1, key_point_1, img2, key_point_2, matches, img_good_matches);
    drawMatches(img1, key_point_1, img2, key_point_2, matches_before_filter, img_matches_before_filter);
    // imshow("matches_before_filter", img_matches_before_filter);
    // imshow("goog_matches", img_good_matches);
    // waitKey(0);
    return img_good_matches;
}

#endif