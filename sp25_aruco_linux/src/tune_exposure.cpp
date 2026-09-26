#include <opencv2/opencv.hpp>
#include "hik_camera.hpp"
#include "robust_aruco_detector.hpp"

#include <algorithm>
#include <deque>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

int main()
{
    HikCamera camera;
    if (!camera.open()) {
        std::cerr << "Camera open failed\n";
        return 1;
    }

    int    exp_auto = 0;
    double exp_us   = 5000.0;
    double gain_db  = 0.0;
    {
        cv::FileStorage fs("config/camera.yaml", cv::FileStorage::READ);
        if (fs.isOpened()) {
            fs["camera"]["exposure_auto"]    >> exp_auto;
            fs["camera"]["exposure_time_us"] >> exp_us;
            fs["camera"]["gain_db"]          >> gain_db;
        }
        fs.release();
    }

    RobustArucoDetector detector;

    const double EXP_STEP  = 500.0;
    const double GAIN_STEP = 1.0;
    const double EXP_MIN   = 50.0;
    const double EXP_MAX   = 1000000.0;
    const double GAIN_MIN  = 0.0;
    const double GAIN_MAX  = 30.0;

    const int WINDOW = 30;
    std::deque<bool> recent;

    auto reset_counter = [&]() { recent.clear(); };

    cv::namedWindow("Tune Exposure", cv::WINDOW_NORMAL);
    std::cout << std::fixed << std::setprecision(1);
    std::cout << "Keys:  +/- exposure  [/] gain  a auto  q quit\n";

    while (true) {
        cv::Mat frame;
        if (!camera.read(frame) || frame.empty()) continue;

        auto detections = detector.detect(frame);

        recent.push_back(!detections.empty());
        while ((int)recent.size() > WINDOW) recent.pop_front();

        int hits = 0;
        for (bool b : recent) if (b) ++hits;
        double rate = recent.empty() ? 0.0 : 100.0 * hits / recent.size();

        std::ostringstream line1;
        line1 << "exp=" << HikCamera::getExposure() << "us"
              << "  set=" << exp_us << "us"
              << "  gain=" << gain_db << "dB"
              << "  auto=" << exp_auto;
        cv::putText(frame, line1.str(), {15, 30},
                    cv::FONT_HERSHEY_SIMPLEX, 0.6,
                    cv::Scalar(0, 255, 255), 2, cv::LINE_AA);

        std::ostringstream line2;
        line2 << "detect: " << hits << "/" << recent.size()
              << "  (" << std::setprecision(0) << rate << "%)";
        cv::putText(frame, line2.str(), {15, 60},
                    cv::FONT_HERSHEY_SIMPLEX, 0.6,
                    cv::Scalar(0, 255, 0), 2, cv::LINE_AA);

        std::string ids_str;
        for (auto& d : detections) ids_str += std::to_string(d.id) + " ";
        cv::putText(frame, "IDs: " + ids_str, {15, 90},
                    cv::FONT_HERSHEY_SIMPLEX, 0.6,
                    cv::Scalar(255, 200, 0), 2, cv::LINE_AA);

        for (auto& d : detections) {
            std::vector<cv::Point> pts;
            for (auto& p : d.corners)
                pts.emplace_back(cvRound(p.x), cvRound(p.y));
            cv::polylines(frame, pts, true, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
        }

        cv::imshow("Tune Exposure", frame);
        int key = cv::waitKey(1);

        if (key == 'q' || key == 'Q' || key == 27) break;

        else if (key == '+' || key == '=') {
            if (exp_auto != 0) { exp_auto = 0; HikCamera::setExposureAuto(0); }
            exp_us = std::min(EXP_MAX, exp_us + EXP_STEP);
            HikCamera::setExposure(exp_us);
            reset_counter();
        }
        else if (key == '-' || key == '_') {
            if (exp_auto != 0) { exp_auto = 0; HikCamera::setExposureAuto(0); }
            exp_us = std::max(EXP_MIN, exp_us - EXP_STEP);
            HikCamera::setExposure(exp_us);
            reset_counter();
        }
        else if (key == ']') {
            gain_db = std::min(GAIN_MAX, gain_db + GAIN_STEP);
            HikCamera::setGain(gain_db);
            reset_counter();
        }
        else if (key == '[') {
            gain_db = std::max(GAIN_MIN, gain_db - GAIN_STEP);
            HikCamera::setGain(gain_db);
            reset_counter();
        }
        else if (key == 'a' || key == 'A') {
            exp_auto = (exp_auto == 0) ? 2 : 0;
            HikCamera::setExposureAuto(exp_auto);
            reset_counter();
        }
    }

    std::cout << "\nFinal: exp=" << exp_us << "us  gain=" << gain_db
              << "dB  auto=" << exp_auto << "\n";
    std::cout << "记得手动改 config/camera.yaml 里的对应字段\n";
    return 0;
}