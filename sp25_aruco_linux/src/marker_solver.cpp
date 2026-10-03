#include "marker_solver.hpp"
#include <opencv2/calib3d.hpp>
#include <cmath>
#include <iostream>
#include <string>
#include <algorithm>

MarkerSolver::MarkerSolver(){
    setDefaults();
    loadConfig("config/camera.yaml");
}

void MarkerSolver::setDefaults(){
    // 硬编码默认值：yaml 读取失败时使用
    K_=(cv::Mat_<double>(3,3)<<
        1746.950650, 0,          706.429157,
        0,           1746.537362, 547.655704,
        0,           0,           1);
    D_=(cv::Mat_<double>(1,5)<<
        -0.073559, 0.102577, 0.000425, -0.000039, 0.0);
    size_ = 0.05;
}

void MarkerSolver::loadConfig(const std::string& path){
    cv::FileStorage fs(path, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cerr << "[Solver] Cannot open " << path
                  << ", using hardcoded defaults\n";
        return;
    }

    double fx=0, fy=0, cx=0, cy=0;
    double k1=0, k2=0, p1=0, p2=0, k3=0;
    double marker_size = size_;

    bool ok = true;

    // 内参
    if (fs["intrinsic"].empty()) {
        std::cerr << "[Solver] yaml has no 'intrinsic'\n";
        ok = false;
    } else {
        fs["intrinsic"]["fx"] >> fx;
        fs["intrinsic"]["fy"] >> fy;
        fs["intrinsic"]["cx"] >> cx;
        fs["intrinsic"]["cy"] >> cy;
    }

    // 畸变
    if (fs["distortion"].empty()) {
        std::cerr << "[Solver] yaml has no 'distortion'\n";
        ok = false;
    } else {
        fs["distortion"]["k1"] >> k1;
        fs["distortion"]["k2"] >> k2;
        fs["distortion"]["p1"] >> p1;
        fs["distortion"]["p2"] >> p2;
        if (!fs["distortion"]["k3"].empty()) {
            fs["distortion"]["k3"] >> k3;
        } else {
            k3 = 0.0;
        }
    }

    // marker 尺寸
    if (!fs["marker_size_m"].empty()) {
        fs["marker_size_m"] >> marker_size;
    }

    fs.release();

    if (!ok) {
        std::cerr << "[Solver] yaml incomplete, using hardcoded defaults\n";
        return;
    }

    K_ = (cv::Mat_<double>(3,3) <<
        fx, 0,  cx,
        0,  fy, cy,
        0,  0,  1);

    D_ = (cv::Mat_<double>(1,5) << k1, k2, p1, p2, k3);

    size_ = marker_size;

    std::cout << "[Solver] Loaded from " << path << "\n"
              << "         fx=" << fx << "  fy=" << fy
              << "  cx=" << cx << "  cy=" << cy << "\n"
              << "         k1=" << k1 << "  k2=" << k2
              << "  p1=" << p1 << "  p2=" << p2
              << "  k3=" << k3 << "\n"
              << "         marker_size=" << size_ << " m\n";
}

Pose MarkerSolver::solve(const Detection& d){
    std::vector<cv::Point3f> obj={
        {-size_/2, size_/2, 0},
        { size_/2, size_/2, 0},
        { size_/2,-size_/2, 0},
        {-size_/2,-size_/2, 0}
    };
    std::vector<cv::Point2f> img(d.corners.begin(), d.corners.end());

    Pose p;
    bool solved = false;
    const char* method = "none";

    // ---- 尝试 1：ITERATIVE + 上一帧初值 ----
    if (has_last_ && last_id_ == d.id) {
        cv::Vec3d rvec = last_rvec_;
        cv::Vec3d tvec = last_tvec_;
        solved = cv::solvePnP(obj, img, K_, D_, rvec, tvec,
                              true, cv::SOLVEPNP_ITERATIVE);
        if (solved) {
            p.rvec = rvec;
            p.tvec = tvec;
            method = "ITER";
        }
    }

    // ---- 回退：IPPE_SQUARE ----
    if (!solved) {
        solved = cv::solvePnP(obj, img, K_, D_, p.rvec, p.tvec,
                              false, cv::SOLVEPNP_IPPE_SQUARE);
        if (solved) method = "IPPE";
    }
/*
    // ================= 诊断打印 =================
    std::cout << "[PnP] id=" << d.id
              << " ok=" << solved
              << " m=" << method
              << " c0=(" << img[0].x << "," << img[0].y << ")"
              << " c1=(" << img[1].x << "," << img[1].y << ")"
              << " c2=(" << img[2].x << "," << img[2].y << ")"
              << " c3=(" << img[3].x << "," << img[3].y << ")";
    // ============================================
*/
    if (!solved) {
        std::cout << "  -> solvePnP failed\n";
        p.valid = false;
        if (++lost_frames_ > 15) has_last_ = false;
        return p;
    }

    p.x = p.tvec[0];
    p.y = p.tvec[1];
    p.z = p.tvec[2];
    p.distance = std::sqrt(p.x*p.x + p.y*p.y + p.z*p.z);

    cv::Mat R;
    cv::Rodrigues(p.rvec, R);
    p.yaw   = std::atan2(R.at<double>(1,0), R.at<double>(0,0)) * 180.0 / CV_PI;
    p.pitch = std::asin(-R.at<double>(2,0))                    * 180.0 / CV_PI;
    p.roll  = std::atan2(R.at<double>(2,1), R.at<double>(2,2)) * 180.0 / CV_PI;

    // 重投影误差
    std::vector<cv::Point2f> proj;
    cv::projectPoints(obj, p.rvec, p.tvec, K_, D_, proj);
    double err = 0;
    for (size_t i = 0; i < proj.size(); ++i) err += cv::norm(proj[i] - img[i]);
    p.reprojection_error = err / proj.size();

    // ---- 按角度动态调阈值 ----
    // 把角度归一化到 [-90, 90] 再取绝对值，避免 ±180 跳变干扰
    auto norm90 = [](double a) {
        while (a >  90.0) a -= 180.0;
        while (a < -90.0) a += 180.0;
        return std::fabs(a);
    };
    double max_angle = std::max({norm90(p.yaw),
                                 norm90(p.pitch),
                                 norm90(p.roll)});

    double eff_max = max_error_;
    if (max_angle > 30.0) {
        // 30° 起线性放宽，60° 时到 2 倍，之后封顶 2 倍
        double extra = (max_angle - 30.0) / 30.0 * max_error_;
        if (extra > max_error_) extra = max_error_;
        eff_max = max_error_ + extra;
    }

    p.valid = (p.reprojection_error <= eff_max);

    // ================= 诊断打印 =================
    /*std::cout << " z=" << p.z
              << " d=" << p.distance
              << " err=" << p.reprojection_error
              << " valid=" << p.valid
              << " angle=" << max_angle
              << " eff_max=" << eff_max
              << "\n";
    // ============================================
*/
    // ---- 更新状态 ----
    if (p.valid) {
        last_rvec_   = p.rvec;
        last_tvec_   = p.tvec;
        last_id_     = d.id;
        has_last_    = true;
        lost_frames_ = 0;
    } else {
        if (++lost_frames_ > 15) has_last_ = false;
    }

    return p;
}