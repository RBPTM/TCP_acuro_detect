#pragma once

#include <opencv2/opencv.hpp>


class FocusEvaluator
{

public:

    static double compute(
        const cv::Mat& image
    );


};