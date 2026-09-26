#include <opencv2/opencv.hpp>
#include "robust_aruco_detector.hpp"
#include "marker_solver.hpp"
#include "hik_camera.hpp"

#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

int main() {
    HikCamera camera;
    if (!camera.open()) {
        std::cerr << "Unable to open Hikrobot camera.\n";
        return 1;
    }

    RobustArucoDetector detector;
    MarkerSolver solver;

    cv::namedWindow("SP25 ArUco", cv::WINDOW_NORMAL);
    std::cout << std::fixed << std::setprecision(3);

    while (true) {
        cv::Mat frame;
        if (!camera.read(frame) || frame.empty()) {
            continue;
        }

        auto detections = detector.detect(frame);
        bool printed = false;

        for (const auto& d : detections) {
            // 画框
            std::vector<cv::Point> pts;
            pts.reserve(4);
            for (const auto& p : d.corners) {
                pts.emplace_back(cvRound(p.x), cvRound(p.y));
            }
            cv::polylines(frame, pts, true, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
            cv::putText(frame, "ID " + std::to_string(d.id), d.center,
                        cv::FONT_HERSHEY_SIMPLEX, 0.7,
                        cv::Scalar(0, 255, 0), 2, cv::LINE_AA);

            // PnP
            Pose pose = solver.solve(d);
            if (!pose.valid) continue;

            // 叠加数据
            std::ostringstream info;
            info << "ID:" << d.id
                 << " Q:" << std::setprecision(2) << d.quality_score
                 << " X:" << std::setprecision(3) << pose.x
                 << " Y:" << pose.y
                 << " Z:" << pose.z;
            cv::putText(frame, info.str(), {15, 30},
                        cv::FONT_HERSHEY_SIMPLEX, 0.6,
                        cv::Scalar(0, 255, 255), 2, cv::LINE_AA);

            std::ostringstream angles;
            angles << "yaw=" << std::setprecision(1) << pose.yaw
                   << " pitch=" << pose.pitch
                   << " roll=" << pose.roll
                   << "  R:" << std::setprecision(2) << pose.reprojection_error << "px";
            cv::putText(frame, angles.str(), {15, 60},
                        cv::FONT_HERSHEY_SIMPLEX, 0.6,
                        cv::Scalar(0, 255, 255), 2, cv::LINE_AA);

            // 终端
            if (!printed) {
                std::cout << "\r"
                          << "ID " << d.id
                          << "  src=" << d.source
                          << "  Q=" << std::setprecision(2) << d.quality_score
                          << "  d=" << std::setprecision(3) << pose.distance << "m"
                          << "  (" << pose.x << "," << pose.y << "," << pose.z << ")"
                          << "  yaw=" << std::setprecision(1) << pose.yaw
                          << "  pitch=" << pose.pitch
                          << "  roll=" << pose.roll
                          << "  err=" << std::setprecision(2) << pose.reprojection_error
                          << "      " << std::flush;
                printed = true;
            }
        }

        cv::imshow("SP25 ArUco", frame);
        int key = cv::waitKey(1);
        if (key == 27 || key == 'q' || key == 'Q') break;
    }

    return 0;
}