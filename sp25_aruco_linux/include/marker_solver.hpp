#pragma once
#include "marker_types.hpp"

class MarkerSolver{
public:
    MarkerSolver();
    Pose solve(const Detection&);

    void setMaxReprojectionError(double e) { max_error_ = e; }

private:
    cv::Mat K_,D_;
    double size_;
    double max_error_ = 3.0;
};