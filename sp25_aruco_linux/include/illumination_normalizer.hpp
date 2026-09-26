#pragma once
#include <opencv2/opencv.hpp>

class IlluminationNormalizer
{
public:
    IlluminationNormalizer();

    void setGaussianSigma(double s) { gaussian_sigma_ = s; }
    void setClaheParams(double clip, int grid)
    {
        clahe_clip_ = clip;
        clahe_grid_ = grid;
    }

    cv::Mat localNormalize(const cv::Mat& image);
    cv::Mat applyCLAHE(const cv::Mat& image);
    cv::Mat gammaCorrect(const cv::Mat& image, double gamma);

private:
    cv::Mat toGray(const cv::Mat& image);

private:
    double gaussian_sigma_ = 15.0;
    double clahe_clip_     = 2.0;
    int    clahe_grid_     = 8;
};