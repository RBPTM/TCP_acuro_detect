
#pragma once
#include <opencv2/opencv.hpp>
#include <array>

struct Detection{
    int id = -1;
    std::array<cv::Point2f,4> corners;
    cv::Point2f center;

    // V3 新增
    double quality_score  = 0.0;
    double border_score   = 0.0;
    double contrast_score = 0.0;
    double geometry_score = 0.0;
    int    source         = 0;
};

struct Pose{
    // 原始 PnP 输出
    cv::Vec3d rvec;
    cv::Vec3d tvec;
    double reprojection_error = 0;
    bool   valid = false;

    // 常用量
    double x = 0, y = 0, z = 0;
    double distance = 0;
    double yaw   = 0;
    double pitch = 0;
    double roll  = 0;
};