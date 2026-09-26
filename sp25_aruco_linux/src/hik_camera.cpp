#include "hik_camera.hpp"
#include "MvCameraControl.h"

#include <cstring>
#include <iostream>
#include <vector>

// 用全局句柄保存相机状态（因为 HikCamera 类没有成员变量）
namespace {
    void* g_handle = nullptr;
    bool  g_grabbing = false;
}

bool HikCamera::open() {
    //MV_CC_Initialize();

    MV_CC_DEVICE_INFO_LIST devList;
    std::memset(&devList, 0, sizeof(devList));

    int ret = MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, &devList);
    if (ret != MV_OK) {
        std::cerr << "[Hik] EnumDevices failed, ret = 0x" << std::hex << ret << std::dec << "\n";
        return false;
    }
    if (devList.nDeviceNum == 0) {
        std::cerr << "[Hik] No camera found\n";
        return false;
    }

    std::cout << "[Hik] Found " << devList.nDeviceNum << " camera(s)\n";

    MV_CC_DEVICE_INFO* devInfo = devList.pDeviceInfo[0];

    ret = MV_CC_CreateHandle(&g_handle, devInfo);
    if (ret != MV_OK) {
        std::cerr << "[Hik] CreateHandle failed, ret = 0x" << std::hex << ret << std::dec << "\n";
        return false;
    }

    ret = MV_CC_OpenDevice(g_handle);
    if (ret != MV_OK) {
        std::cerr << "[Hik] OpenDevice failed, ret = 0x" << std::hex << ret << std::dec << "\n";
        MV_CC_DestroyHandle(g_handle);
        g_handle = nullptr;
        return false;
    }

    // ---- 从 config/camera.yaml 读曝光设置 ----
    int    exp_auto = 0;
    double exp_us   = 5000.0;
    double gain_db  = 0.0;

    cv::FileStorage fs("config/camera.yaml", cv::FileStorage::READ);
    if (fs.isOpened()) {
        fs["camera"]["exposure_auto"]    >> exp_auto;
        fs["camera"]["exposure_time_us"] >> exp_us;
        fs["camera"]["gain_db"]          >> gain_db;
    } else {
        std::cerr << "[Hik] Cannot open config/camera.yaml, use defaults\n";
    }
    fs.release();

    MV_CC_SetEnumValue(g_handle, "ExposureAuto", exp_auto);
    MV_CC_SetFloatValue(g_handle, "ExposureTime", (float)exp_us);
    MV_CC_SetFloatValue(g_handle, "Gain",         (float)gain_db);

    std::cout << "[Hik] Exposure: auto=" << exp_auto
              << "  time=" << exp_us << "us"
              << "  gain=" << gain_db << "dB\n";

    MV_CC_SetEnumValue(g_handle, "TriggerMode", 0);
    MV_CC_SetEnumValue(g_handle, "PixelFormat", PixelType_Gvsp_BGR8_Packed);

    ret = MV_CC_StartGrabbing(g_handle);
    if (ret != MV_OK) {
        std::cerr << "[Hik] StartGrabbing failed, ret = 0x" << std::hex << ret << std::dec << "\n";
        MV_CC_CloseDevice(g_handle);
        MV_CC_DestroyHandle(g_handle);
        g_handle = nullptr;
        return false;
    }

    g_grabbing = true;
    std::cout << "[Hik] Camera opened and grabbing\n";
    return true;
}

bool HikCamera::read(cv::Mat& out) {
    if (!g_handle || !g_grabbing) return false;

    MV_FRAME_OUT stImageInfo;
    std::memset(&stImageInfo, 0, sizeof(stImageInfo));

    int ret = MV_CC_GetImageBuffer(g_handle, &stImageInfo, 1000);
    if (ret != MV_OK) return false;

    const unsigned int w  = stImageInfo.stFrameInfo.nWidth;
    const unsigned int h  = stImageInfo.stFrameInfo.nHeight;
    const unsigned int px = stImageInfo.stFrameInfo.enPixelType;
    unsigned char* data   = stImageInfo.pBufAddr;

    if (px == PixelType_Gvsp_Mono8) {
        cv::Mat gray(h, w, CV_8UC1, data);
        cv::cvtColor(gray, out, cv::COLOR_GRAY2BGR);
    }
    else if (px == PixelType_Gvsp_BGR8_Packed) {
        cv::Mat tmp(h, w, CV_8UC3, data);
        out = tmp.clone();
    }
    else if (px == PixelType_Gvsp_RGB8_Packed) {
        cv::Mat tmp(h, w, CV_8UC3, data);
        cv::cvtColor(tmp, out, cv::COLOR_RGB2BGR);
    }
    else {
        static std::vector<unsigned char> cvtBuf;
        cvtBuf.resize(static_cast<size_t>(w) * h * 3);

        MV_CC_PIXEL_CONVERT_PARAM cvtParam;
        std::memset(&cvtParam, 0, sizeof(cvtParam));
        cvtParam.nWidth         = w;
        cvtParam.nHeight        = h;
        cvtParam.pSrcData       = data;
        cvtParam.nSrcDataLen    = stImageInfo.stFrameInfo.nFrameLen;
        cvtParam.enSrcPixelType = static_cast<MvGvspPixelType>(px);
        cvtParam.enDstPixelType = PixelType_Gvsp_BGR8_Packed;
        cvtParam.pDstBuffer     = cvtBuf.data();
        cvtParam.nDstBufferSize = static_cast<unsigned int>(cvtBuf.size());

        int cvtRet = MV_CC_ConvertPixelType(g_handle, &cvtParam);
        if (cvtRet != MV_OK) {
            MV_CC_FreeImageBuffer(g_handle, &stImageInfo);
            return false;
        }

        cv::Mat tmp(h, w, CV_8UC3, cvtBuf.data());
        out = tmp.clone();
    }

    MV_CC_FreeImageBuffer(g_handle, &stImageInfo);
    return true;
}

// ---- 新增：运行时曝光/增益控制 ----
void HikCamera::setExposure(double us) {
    if (!g_handle) return;
    MV_CC_SetFloatValue(g_handle, "ExposureTime", (float)us);
}

void HikCamera::setGain(double db) {
    if (!g_handle) return;
    MV_CC_SetFloatValue(g_handle, "Gain", (float)db);
}

void HikCamera::setExposureAuto(int mode) {
    if (!g_handle) return;
    MV_CC_SetEnumValue(g_handle, "ExposureAuto", mode);
}

double HikCamera::getExposure() {
    if (!g_handle) return 0.0;
    MVCC_FLOATVALUE v;
    std::memset(&v, 0, sizeof(v));
    if (MV_CC_GetFloatValue(g_handle, "ExposureTime", &v) != MV_OK) return 0.0;
    return v.fCurValue;
}