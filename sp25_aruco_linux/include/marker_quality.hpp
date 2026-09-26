#pragma once
#include <opencv2/opencv.hpp>
#include "marker_types.hpp"

class MarkerQuality
{
public:
    // 边框对比度：marker 内侧黑边 vs 外侧白区
    static double borderScore(const Detection& det, const cv::Mat& image);

    // 内部对比度：marker 内部区域灰度标准差
    static double contrastScore(const Detection& det, const cv::Mat& image);

    // 几何分数：四条边长是否一致
    static double geometryScore(const Detection& det);

    // 加权总分
    static double totalScore(const Detection& det,
                             const cv::Mat& image,
                             double w_border   = 0.4,
                             double w_contrast = 0.3,
                             double w_geometry = 0.3);
};