#pragma once
#include <opencv2/aruco.hpp>
#include "marker_types.hpp"

class ArucoDetector{
public:
    ArucoDetector();
    explicit ArucoDetector(const cv::aruco::DetectorParameters& params);

    std::vector<Detection> detect(const cv::Mat& frame);

private:
    cv::Ptr<cv::aruco::Dictionary> dict_;
    cv::Ptr<cv::aruco::DetectorParameters> params_;
};