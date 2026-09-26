#pragma once
#include <opencv2/aruco.hpp>
#include <string>
#include <vector>

#include "marker_types.hpp"
#include "illumination_normalizer.hpp"

class RobustArucoDetector
{
public:
    RobustArucoDetector();
    explicit RobustArucoDetector(const std::string& configPath);

    std::vector<Detection> detect(const cv::Mat& image);

private:
    struct BranchResult {
        std::vector<Detection> detections;
        int source;
    };

    void loadConfig(const std::string& path);
    void setupDefaults();

    std::vector<Detection> detectSingle(const cv::Mat& image,
                                        const cv::Ptr<cv::aruco::DetectorParameters>& params,
                                        int source);

    std::vector<Detection> fuse(const std::vector<BranchResult>& all);

private:
    cv::Ptr<cv::aruco::Dictionary> dict_;

    cv::Ptr<cv::aruco::DetectorParameters> normal_params_;
    cv::Ptr<cv::aruco::DetectorParameters> shadow_params_;
    cv::Ptr<cv::aruco::DetectorParameters> low_contrast_params_;

    IlluminationNormalizer normalizer_;

    bool   enable_local_normalization_ = true;
    bool   enable_clahe_               = true;
    bool   enable_gamma_               = true;
    double gaussian_sigma_             = 15.0;
    double clahe_clip_                 = 2.0;
    int    clahe_grid_                 = 8;
    double gamma_dark_                 = 0.7;
    double gamma_bright_               = 1.3;

    double min_border_score_   = 0.55;
    double min_contrast_score_ = 0.10;
    double min_total_score_    = 0.60;
};