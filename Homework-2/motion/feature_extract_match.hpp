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
实现两张图像的特征提取和匹配：
使用引用的目的是为了直接将提取和匹配的结果保存在key_point_1，key_point_2，matches中
*/
Mat feature_extract_match(Mat img1, Mat img2, vector<KeyPoint>& key_point_1, vector<KeyPoint>& key_point_2, vector<DMatch>& matches)
{

    //创建ORB提取器、描述子提取器、描述子匹配器
    Ptr<FeatureDetector> orb_detectore = ORB::create();
    Ptr<DescriptorExtractor> orb_descriptor = ORB::create();
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");

    //定义变量
    Mat descriptor_1, descriptor_2;
    vector<DMatch> matches_before_filter;
    Mat img_matches_before_filter, img_good_matches;

    //提取ORB关键点
    orb_detectore->detect(img1, key_point_1);
    orb_detectore->detect(img2, key_point_2);

    //计算描述子
    orb_descriptor->compute(img1, key_point_1, descriptor_1);
    orb_descriptor->compute(img2, key_point_2, descriptor_2);

    //特征点匹配，匹配结果中会包含一些误匹配的点
    matcher->match(descriptor_1, descriptor_2, matches_before_filter);

    //剔除误匹配的点：汉明距离大于两倍最小距离的点剔除
    double min_distance = 9999;
    for (int i = 0; i < descriptor_1.rows; i++)
    {
        if (matches_before_filter[i].distance < min_distance)
        {
            min_distance = matches_before_filter[i].distance;
        }
    }
    cout << "min_distance = " << min_distance << endl;

    //有的时候最小距离非常小，这个情况下取最小距离为一个经验值30
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