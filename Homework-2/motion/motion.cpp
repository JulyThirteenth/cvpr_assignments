#include "estimation.hpp"
#include "feature_extract_match.hpp"

//opencv，warpPerspective()在这个头文件中
#include <opencv2/imgproc.hpp>

void multipleImage(vector<Mat> imgVector, Mat& dst, int imgCols)
{
    const int MAX_PIXEL = 300;
    int imgNum = imgVector.size();
    //选择图片最大的一边 将最大的边按比例变为300像素
    Size imgOriSize = imgVector[0].size();
    int imgMaxPixel = max(imgOriSize.height, imgOriSize.width);
    //获取最大像素变为MAX_PIXEL的比例因子
    double prop = imgMaxPixel < MAX_PIXEL ? (double)imgMaxPixel / MAX_PIXEL : MAX_PIXEL / (double)imgMaxPixel;
    Size imgStdSize(imgOriSize.width * prop, imgOriSize.height * prop); //窗口显示的标准图像的Size

    Mat imgStd; //标准图片
    Point2i location(0, 0); //坐标点(从0,0开始)
    //构建窗口大小 通道与imageVector[0]的通道一样
    Mat imgWindow(imgStdSize.height * ((imgNum - 1) / imgCols + 1), imgStdSize.width * imgCols, imgVector[0].type());
    for (int i = 0; i < imgNum; i++)
    {
        location.x = (i % imgCols) * imgStdSize.width;
        location.y = (i / imgCols) * imgStdSize.height;
        resize(imgVector[i], imgStd, imgStdSize, prop, prop, INTER_LINEAR); //设置为标准大小
        imgStd.copyTo(imgWindow(Rect(location, imgStdSize)));
    }
    dst = imgWindow;
}

int main(int argc, char** argv)
{
    VideoCapture cap("D:/Users/Robot/DevFiles/cvpr_assignments/Homework-2/GOPR0110.MP4");
    Mat frame1, frame2;
    cap >> frame1;
    cap >> frame2;
    while (!frame1.empty() && !frame2.empty())
    {
        vector<KeyPoint> key_points_1, key_points_2;
        vector<DMatch> matches;
        Mat res = feature_extract_match(frame1, frame2, key_points_1, key_points_2, matches);
        cout << "一共找到了： " << matches.size() << "个匹配点" << endl;

        Mat R, t;
        pose_estimation_2d2d(key_points_1, key_points_2, matches, R, t);
        cout << "R = " << endl;
        cout << R << endl;
        cout << "t = " << endl;
        cout << t << endl;

        //Mat H;
        //Mat frame_warp_perspective;
        //homograph_estimation(key_points_1, key_points_2, matches, H);
        //warpPerspective(frame1, frame_warp_perspective, H, frame2.size());    //利用单应矩阵进行转换
        
        vector<Mat> imgVector;
        imgVector.push_back(frame1);
        imgVector.push_back(frame2);
        imgVector.push_back(res);
        //imgVector.push_back(frame_warp_perspective);
        Mat dst;
        multipleImage(imgVector, dst, 3);
        namedWindow("multipleWindow");
        imshow("multipleWindow", dst);
        waitKey(1);
        frame1 = frame2;
        cap >> frame2;
    }
    return 0;
}