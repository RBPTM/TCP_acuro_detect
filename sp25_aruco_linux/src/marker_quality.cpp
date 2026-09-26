#include "marker_quality.hpp"
#include <algorithm>
#include <cmath>

static inline uchar sampleGray(const cv::Mat& gray, const cv::Point2f& p)
{
    int x = cvRound(p.x);
    int y = cvRound(p.y);
    x = std::max(0, std::min(gray.cols - 1, x));
    y = std::max(0, std::min(gray.rows - 1, y));
    return gray.at<uchar>(y, x);
}

static cv::Mat toGray(const cv::Mat& image)
{
    cv::Mat gray;
    if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = image.clone();
    }
    return gray;
}

double MarkerQuality::borderScore(const Detection& det, const cv::Mat& image)
{
    cv::Mat gray = toGray(image);

    cv::Point2f center(0.f, 0.f);
    for (const auto& c : det.corners) center += c;
    center *= 0.25f;

    double sum = 0.0;
    for (int i = 0; i < 4; ++i) {
        const cv::Point2f& c = det.corners[i];
        cv::Point2f dir = c - center;

        // 内点：往中心缩 8%
        cv::Point2f p_in  = c - dir * 0.08f;
        // 外点：往外扩 12%
        cv::Point2f p_out = c + dir * 0.12f;

        double g_in  = sampleGray(gray, p_in);
        double g_out = sampleGray(gray, p_out);

        // 外白内黑 -> 正差值
        sum += (g_out - g_in) / 255.0;
    }
    double score = sum / 4.0;
    return std::max(0.0, std::min(1.0, score));
}

double MarkerQuality::contrastScore(const Detection& det, const cv::Mat& image)
{
    cv::Mat gray = toGray(image);

    // 用角点 bounding box 的内缩区域
    cv::Rect bb = cv::boundingRect(det.corners);
    int mx = static_cast<int>(bb.width  * 0.20);
    int my = static_cast<int>(bb.height * 0.20);
    int x0 = std::max(0, bb.x + mx);
    int y0 = std::max(0, bb.y + my);
    int x1 = std::min(gray.cols, bb.x + bb.width  - mx);
    int y1 = std::min(gray.rows, bb.y + bb.height - my);

    if (x1 - x0 < 4 || y1 - y0 < 4) return 0.0;

    cv::Mat roi = gray(cv::Rect(x0, y0, x1 - x0, y1 - y0));

    cv::Scalar mean, stddev;
    cv::meanStdDev(roi, mean, stddev);

    double score = stddev[0] / 128.0;
    return std::max(0.0, std::min(1.0, score));
}

double MarkerQuality::geometryScore(const Detection& det)
{
    double l[4];
    for (int i = 0; i < 4; ++i) {
        const cv::Point2f& a = det.corners[i];
        const cv::Point2f& b = det.corners[(i + 1) % 4];
        l[i] = cv::norm(b - a);
    }

    double mean = (l[0] + l[1] + l[2] + l[3]) / 4.0;
    if (mean < 1e-6) return 0.0;

    double var = 0.0;
    for (int i = 0; i < 4; ++i) var += (l[i] - mean) * (l[i] - mean);
    var /= 4.0;
    double stddev = std::sqrt(var);

    double score = 1.0 - stddev / mean;
    return std::max(0.0, std::min(1.0, score));
}

double MarkerQuality::totalScore(const Detection& det,
                                 const cv::Mat& image,
                                 double w_border,
                                 double w_contrast,
                                 double w_geometry)
{
    double b = borderScore(det, image);
    double c = contrastScore(det, image);
    double g = geometryScore(det);

    double wsum = w_border + w_contrast + w_geometry;
    if (wsum < 1e-6) return 0.0;

    return (w_border * b + w_contrast * c + w_geometry * g) / wsum;
}