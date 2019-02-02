#include <gtest/gtest.h>
#include "posest/blurrer.h"
#include "posest/reprojector.h"
#include "posest/dataset.h"

using namespace posest;

TEST(blurrer, blur) {
    Dataset dataset("/realistic-dataset");
    dataset.read();

    const cv::Mat_<uchar> &ref_sharp = dataset.readSharpImage(1, 0);
    const cv::Mat_<double> &ref_depth = dataset.readDepthImage(1, 0);
    const cv::Mat_<uchar> &blurred_img = dataset.readBlurredImage(2, 0);
    const mrpt::poses::CPose3DQuat &ref_pose = dataset.getPose(1, 0);
    const InternalCalibration &internalCalibration = dataset.getInternalCalibration(0);
    const double exposure_time = dataset.getExposureTime();

    posest::ReprojectorImpl reprojector(internalCalibration, ref_sharp, ref_depth, ref_pose);

    //TODO: set time of reference pose and end time
    double ref_time = 0.1, end_time = 0.2;
    std::cout << "Exposure time: " << exposure_time << std::endl;

    posest::BlurrerImpl blurrer(dataset.getPose(1, 0), ref_time, end_time, exposure_time, 8, ref_sharp, reprojector);
    mrpt::poses::CPose3DQuat end_pose = dataset.getPoseAtTime(0.2, 0);

    cv::Mat_<uchar> blurred_img_syntethic;
    blurrer.blur(end_pose, blurred_img_syntethic);
    cv::imwrite("blurred_img_synthetic.png", blurred_img_syntethic);
}

