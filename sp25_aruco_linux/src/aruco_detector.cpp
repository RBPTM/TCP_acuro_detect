#include "aruco_detector.hpp"

ArucoDetector::ArucoDetector(){
    dict_ = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50);
    params_ = cv::aruco::DetectorParameters::create();
    params_->cornerRefinementMethod = cv::aruco::CORNER_REFINE_SUBPIX;
}

ArucoDetector::ArucoDetector(const cv::aruco::DetectorParameters& params){
    dict_ = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50);
    params_ = cv::aruco::DetectorParameters::create();
    *params_ = params;
    params_->cornerRefinementMethod = cv::aruco::CORNER_REFINE_SUBPIX;
}

std::vector<Detection> ArucoDetector::detect(const cv::Mat& frame){
    std::vector<int> ids;
    std::vector<std::vector<cv::Point2f>> c, r;
    cv::aruco::detectMarkers(frame, dict_, c, ids, params_, r);

    std::vector<Detection> out;
    for(size_t i=0;i<ids.size();i++){
        Detection d; d.id=ids[i];
        for(int j=0;j<4;j++) d.corners[j]=c[i][j];
        d.center=(c[i][0]+c[i][2])*0.5f;
        out.push_back(d);
    }
    return out;
}