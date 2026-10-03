#pragma once
#include "marker_types.hpp"
#include <string>

class MarkerSolver{
public:
    MarkerSolver();
    Pose solve(const Detection&);

    void setMaxReprojectionError(double e) { max_error_ = e; }

private:
    void loadConfig(const std::string& path);
    void setDefaults();

    cv::Mat K_,D_;
    double size_;
    double max_error_ = 3.0;

    // Plan B: 上一帧位姿作为 ITERATIVE 初值
    cv::Vec3d last_rvec_{0,0,0};
    cv::Vec3d last_tvec_{0,0,0};
    bool      has_last_    = false;
    int       last_id_     = -1;
    int       lost_frames_ = 0;
};