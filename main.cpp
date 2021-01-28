#include <iostream>
#include <opencv2/opencv.hpp>
/*
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>
*/

using namespace cv;
using namespace std;

bool selectObject = false; // 用于标记是否有选取目标
int trackObject = 0;       // 1 表示有追踪对象 0 表示无追踪对象 -1 表示追踪对象尚未计算 Camshift 所需的属性
Rect selection;            // 保存鼠标选择的区域
Mat image;                 // 用于缓存读取到的视频帧

void onMouse(int event, int x, int y, int, void*) 
{
    static Point origin;
    if (selectObject) 
    {
        // 确定鼠标选定区域的左上角坐标以及区域的长和宽
        selection.x = MIN(x, origin.x);
        selection.y = MIN(y, origin.y);
        selection.width = abs(x - origin.x);
        selection.height = abs(y - origin.y);
        // & 运算符被 cv::Rect 重载
        // 表示两个区域取交集, 主要目的是为了处理当鼠标在选择区域时移除画面外
        selection &= Rect(0, 0, image.cols, image.rows);
    }
    switch (event)
    {
        // 处理鼠标左键被按下
    case EVENT_LBUTTONDOWN:
        origin = Point(x, y);
        selection = Rect(x, y, 0, 0);
        selectObject = true;
        break;
        // 处理鼠标左键被抬起
    case EVENT_LBUTTONUP:
        selectObject = false;
        if (selection.width > 0 && selection.height > 0)
            trackObject = -1; // 追踪的目标还未计算 Camshift 所需要的属性
        break;
    }
}

int main(int argc, char** argv)
{
    VideoCapture video(0);
    namedWindow("CamshiftOBJTracking");

    setMouseCallback("CamshiftOBJTracking", onMouse, 0);
    
    Mat frame, hsv, hue, mask, hist, backproj;
    Rect trackWindow;

    int hsize = 16;
    float hranges[] = { 0,180 };
    const float* phranges = hranges;

    while (true) 
    {
        video >> frame;
        if (frame.empty())
            break;
        frame.copyTo(image);
        cvtColor(image, hsv, COLOR_BGR2HSV);
        
        if (trackObject) 
        {
            inRange(hsv, Scalar(0, 30, 10), Scalar(180, 256, 256), mask); //制作掩模版
            int ch[] = { 0, 0 };
            hue.create(hsv.size(), hsv.depth());
            mixChannels(&hsv, 1, &hue, 1, ch, 1);
            
            if (trackObject < 0) 
            {
                Mat roi(hue, selection), maskroi(mask, selection);
                calcHist(&roi, 1, 0, maskroi, hist, 1, &hsize, &phranges);
                normalize(hist, hist, 0, 255, NORM_MINMAX);
                trackWindow = selection;
                trackObject = 1;
            }
            calcBackProject(&hue, 1, 0, hist, backproj, &phranges);
            backproj &= mask;
            RotatedRect trackBox = CamShift(backproj, trackWindow, TermCriteria(TermCriteria::EPS | TermCriteria::COUNT, 10, 1));
            if (trackWindow.area() <= 1) 
            {
                int cols = backproj.cols, rows = backproj.rows, r = (MIN(cols, rows) + 5) / 6;
                trackWindow = Rect(trackWindow.x - r, trackWindow.y - r, trackWindow.x + r, trackWindow.y + r) & Rect(0, 0, cols, rows);
            }
            ellipse(image, trackBox, Scalar(0, 0, 255), 3);
        }
        if (selectObject && selection.width > 0 && selection.height > 0) {
            Mat roi(image, selection);
            bitwise_not(roi, roi);
        }
        
        imshow("CamshiftOBJTracking", image);
        
        int keyboard = waitKey(30);
        if (keyboard == 'q' || keyboard == 27)
            break;
    }
    destroyAllWindows();
    video.release();
    return 0;
}