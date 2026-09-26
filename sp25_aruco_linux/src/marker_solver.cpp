#include "marker_solver.hpp"
#include <opencv2/calib3d.hpp>
#include <cmath>

MarkerSolver::MarkerSolver(){
    K_=(cv::Mat_<double>(3,3)<<1773.539686,0,706.740881,0,1774.272085,547.274422,0,0,1);
    D_=(cv::Mat_<double>(1,5)<<-0.072105571,0.099650345,0.000675799,0.000037105,0);
    size_=0.05;
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
    cv::solvePnP(obj, img, K_, D_, p.rvec, p.tvec, false, cv::SOLVEPNP_IPPE_SQUARE);

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

    // 质量过滤
    p.valid = (p.reprojection_error <= max_error_);

    return p;
}