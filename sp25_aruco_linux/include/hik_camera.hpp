
#pragma once
#include <opencv2/opencv.hpp>
class HikCamera{
public:
 bool open();
 bool read(cv::Mat&);

// 新增：供测试程序调节使用
  static void   setExposure(double us);
  static void   setGain(double db);
  static void   setExposureAuto(int mode);
  static double getExposure();
};
