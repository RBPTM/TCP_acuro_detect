#include "robust_aruco_detector.hpp"
#include "marker_quality.hpp"

#include <iostream>

RobustArucoDetector::RobustArucoDetector()
{
    dict_ = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50);
    setupDefaults();
    loadConfig("config/aruco.yaml");
}

RobustArucoDetector::RobustArucoDetector(const std::string& configPath)
{
    dict_ = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50);
    setupDefaults();
    loadConfig(configPath);
}

void RobustArucoDetector::setupDefaults()
{
    normal_params_ = cv::aruco::DetectorParameters::create();
    normal_params_->cornerRefinementMethod = cv::aruco::CORNER_REFINE_SUBPIX;

    shadow_params_ = cv::aruco::DetectorParameters::create();
    shadow_params_->adaptiveThreshWinSizeMin  = 5;
    shadow_params_->adaptiveThreshWinSizeMax  = 41;
    shadow_params_->adaptiveThreshWinSizeStep = 6;
    shadow_params_->adaptiveThreshConstant    = 5;
    shadow_params_->cornerRefinementMethod    = cv::aruco::CORNER_REFINE_SUBPIX;

    low_contrast_params_ = cv::aruco::DetectorParameters::create();
    low_contrast_params_->adaptiveThreshWinSizeMin  = 7;
    low_contrast_params_->adaptiveThreshWinSizeMax  = 51;
    low_contrast_params_->adaptiveThreshWinSizeStep = 6;
    low_contrast_params_->adaptiveThreshConstant    = 9;
    low_contrast_params_->cornerRefinementMethod    = cv::aruco::CORNER_REFINE_SUBPIX;
}

void RobustArucoDetector::loadConfig(const std::string& path)
{
    cv::FileStorage fs(path, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cerr << "[Robust] Cannot open " << path << ", using defaults\n";
        return;
    }

    // ---- illumination ----
    int flag = 0;
    double d = 0.0;

    if (!fs["illumination"].empty()) {
        fs["illumination"]["enable_local_normalization"] >> flag;
        enable_local_normalization_ = (flag != 0);

        if (!fs["illumination"]["gaussian_sigma"].empty()) {
            fs["illumination"]["gaussian_sigma"] >> d;
            gaussian_sigma_ = d;
        }

        fs["illumination"]["enable_clahe"] >> flag;
        enable_clahe_ = (flag != 0);

        if (!fs["illumination"]["clahe_clip_limit"].empty()) {
            fs["illumination"]["clahe_clip_limit"] >> d;
            clahe_clip_ = d;
        }
        if (!fs["illumination"]["clahe_grid_size"].empty()) {
            fs["illumination"]["clahe_grid_size"] >> flag;
            clahe_grid_ = flag;
        }

        fs["illumination"]["enable_gamma"] >> flag;
        enable_gamma_ = (flag != 0);

        if (!fs["illumination"]["gamma_dark"].empty()) {
            fs["illumination"]["gamma_dark"] >> d;
            gamma_dark_ = d;
        }
        if (!fs["illumination"]["gamma_bright"].empty()) {
            fs["illumination"]["gamma_bright"] >> d;
            gamma_bright_ = d;
        }
    }

    normalizer_.setGaussianSigma(gaussian_sigma_);
    normalizer_.setClaheParams(clahe_clip_, clahe_grid_);

    // ---- quality ----
    if (!fs["quality"].empty()) {
        if (!fs["quality"]["min_border_score"].empty()) {
            fs["quality"]["min_border_score"] >> d;
            min_border_score_ = d;
        }
        if (!fs["quality"]["min_contrast_score"].empty()) {
            fs["quality"]["min_contrast_score"] >> d;
            min_contrast_score_ = d;
        }
        if (!fs["quality"]["min_total_score"].empty()) {
            fs["quality"]["min_total_score"] >> d;
            min_total_score_ = d;
        }
    }

    // ---- detector params ----
    auto read_params = [&](const std::string& name,
                           cv::Ptr<cv::aruco::DetectorParameters>& out) {
        if (fs["detector"][name].empty()) return;
        fs["detector"][name]["adaptive_thresh_win_size_min"]  >> d;
        out->adaptiveThreshWinSizeMin  = static_cast<int>(d);
        fs["detector"][name]["adaptive_thresh_win_size_max"]  >> d;
        out->adaptiveThreshWinSizeMax  = static_cast<int>(d);
        fs["detector"][name]["adaptive_thresh_win_size_step"] >> d;
        out->adaptiveThreshWinSizeStep = static_cast<int>(d);
        fs["detector"][name]["adaptive_thresh_constant"]      >> d;
        out->adaptiveThreshConstant    = d;
    };

    read_params("normal",       normal_params_);
    read_params("shadow",       shadow_params_);
    read_params("low_contrast", low_contrast_params_);

    fs.release();

    std::cout << "[Robust] Config loaded: localnorm=" << enable_local_normalization_
              << "  clahe=" << enable_clahe_
              << "  gamma=" << enable_gamma_ << "\n";
}

std::vector<Detection>
RobustArucoDetector::detectSingle(const cv::Mat& image,
                                  const cv::Ptr<cv::aruco::DetectorParameters>& params,
                                  int source)
{
    std::vector<int> ids;
    std::vector<std::vector<cv::Point2f>> corners;
    std::vector<std::vector<cv::Point2f>> rejected;

    cv::aruco::detectMarkers(image, dict_, corners, ids, params, rejected);

    std::vector<Detection> out;
    out.reserve(ids.size());

    for (size_t i = 0; i < ids.size(); ++i) {
        if (corners[i].size() != 4) continue;

        Detection det;
        det.id = ids[i];
        for (int j = 0; j < 4; ++j) det.corners[j] = corners[i][j];
        det.center = (corners[i][0] + corners[i][2]) * 0.5f;
        det.source = source;

        det.border_score   = MarkerQuality::borderScore(det, image);
        det.contrast_score = MarkerQuality::contrastScore(det, image);
        det.geometry_score = MarkerQuality::geometryScore(det);
        det.quality_score  = MarkerQuality::totalScore(det, image);

        // 质量门槛
        if (det.border_score   < min_border_score_)   continue;
        if (det.contrast_score < min_contrast_score_) continue;
        if (det.quality_score  < min_total_score_)    continue;

        out.push_back(det);
    }
    return out;
}

std::vector<Detection>
RobustArucoDetector::detect(const cv::Mat& image)
{
    std::vector<BranchResult> all;

    // 分支 0：原图 + normal
    all.push_back({ detectSingle(image, normal_params_, 0), 0 });
    // 分支 0：原图 + shadow
    all.push_back({ detectSingle(image, shadow_params_, 0), 0 });
    // 分支 0：原图 + low_contrast
    all.push_back({ detectSingle(image, low_contrast_params_, 0), 0 });

    // 分支 1：局部归一化 + normal
    if (enable_local_normalization_) {
        cv::Mat ln = normalizer_.localNormalize(image);
        all.push_back({ detectSingle(ln, normal_params_, 1), 1 });
    }

    // 分支 2：CLAHE + normal
    if (enable_clahe_) {
        cv::Mat cl = normalizer_.applyCLAHE(image);
        all.push_back({ detectSingle(cl, normal_params_, 2), 2 });
    }

    // 分支 3：Gamma + normal
    if (enable_gamma_) {
        cv::Mat gd = normalizer_.gammaCorrect(image, gamma_dark_);
        all.push_back({ detectSingle(gd, normal_params_, 3), 3 });

        cv::Mat gb = normalizer_.gammaCorrect(image, gamma_bright_);
        all.push_back({ detectSingle(gb, normal_params_, 3), 3 });
    }

    return fuse(all);
}

std::vector<Detection>
RobustArucoDetector::fuse(const std::vector<BranchResult>& all)
{
    // 同一 ID 保留 quality_score 最高的候选
    std::vector<Detection> best;
    std::vector<double>    bestScore;

    for (const auto& br : all) {
        for (const auto& d : br.detections) {
            bool merged = false;
            for (size_t i = 0; i < best.size(); ++i) {
                if (best[i].id == d.id) {
                    if (d.quality_score > bestScore[i]) {
                        best[i]      = d;
                        bestScore[i] = d.quality_score;
                    }
                    merged = true;
                    break;
                }
            }
            if (!merged) {
                best.push_back(d);
                bestScore.push_back(d.quality_score);
            }
        }
    }
    return best;
}