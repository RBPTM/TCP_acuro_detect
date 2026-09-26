#include "illumination_normalizer.hpp"

IlluminationNormalizer::IlluminationNormalizer() {}

cv::Mat IlluminationNormalizer::toGray(const cv::Mat& image)
{
    cv::Mat gray;
    if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = image.clone();
    }
    return gray;
}

cv::Mat IlluminationNormalizer::localNormalize(const cv::Mat& image)
{
    cv::Mat gray = toGray(image);

    // 估计光照：低通滤波
    cv::Mat illum;
    cv::GaussianBlur(gray, illum, cv::Size(0, 0), gaussian_sigma_);

    // 逐像素除法（防除零）
    cv::Mat gray_f, illum_f;
    gray.convertTo(gray_f, CV_32F);
    illum.convertTo(illum_f, CV_32F);
    illum_f += 1.0f;

    cv::Mat result_f = gray_f * 128.0f / illum_f;

    cv::Mat result;
    result_f.convertTo(result, CV_8U);

    // 拉伸到 0-255
    cv::normalize(result, result, 0, 255, cv::NORM_MINMAX);

    cv::Mat bgr;
    cv::cvtColor(result, bgr, cv::COLOR_GRAY2BGR);
    return bgr;
}

cv::Mat IlluminationNormalizer::applyCLAHE(const cv::Mat& image)
{
    cv::Mat gray = toGray(image);

    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(
        clahe_clip_,
        cv::Size(clahe_grid_, clahe_grid_));
    clahe->apply(gray, gray);

    cv::Mat bgr;
    cv::cvtColor(gray, bgr, cv::COLOR_GRAY2BGR);
    return bgr;
}

cv::Mat IlluminationNormalizer::gammaCorrect(const cv::Mat& image, double gamma)
{
    cv::Mat gray = toGray(image);

    // 查表加速
    cv::Mat lut(1, 256, CV_8U);
    uchar* p = lut.ptr<uchar>(0);
    for (int i = 0; i < 256; ++i) {
        double v = std::pow(i / 255.0, gamma) * 255.0;
        v = std::min(255.0, std::max(0.0, v));
        p[i] = static_cast<uchar>(v);
    }

    cv::Mat out;
    cv::LUT(gray, lut, out);

    cv::Mat bgr;
    cv::cvtColor(out, bgr, cv::COLOR_GRAY2BGR);
    return bgr;
}