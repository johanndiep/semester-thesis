#include "gtest/gtest.h"
#include "posest/dataset.h"

using namespace posest;
using namespace std;


class DatasetTests : public ::testing::Test {

 protected:
    Dataset *dataset;

    virtual void SetUp() {
        // TODO(tiaebi): make this configurable with cmake
        const string path("/realistic-dataset");
        dataset = new Dataset(path);
        dataset->read();
    }

    virtual void TearDown() {
        delete dataset;
    }
};

TEST_F(DatasetTests, getCameras) {
    const vector<string> &cams = dataset->getCameras();
    ASSERT_EQ(cams.size(), 2);
    EXPECT_EQ(cams[0], "cam0");
    EXPECT_EQ(cams[1], "cam1");
}

TEST_F(DatasetTests, getInternalCalibration) {
    const InternalCalibration &ic0 = dataset->getInternalCalibration(0);
    EXPECT_DOUBLE_EQ(320, ic0.fx);
    EXPECT_DOUBLE_EQ(320, ic0.fy);
    EXPECT_DOUBLE_EQ(320, ic0.cx);
    EXPECT_DOUBLE_EQ(240, ic0.cy);

    const InternalCalibration &ic1 = dataset->getInternalCalibration(1);
    EXPECT_DOUBLE_EQ(320, ic1.fx);
    EXPECT_DOUBLE_EQ(320, ic1.fy);
    EXPECT_DOUBLE_EQ(320, ic1.cx);
    EXPECT_DOUBLE_EQ(240, ic1.cy);
}

TEST_F(DatasetTests, readDepthImage) {
    // TODO(tiaebi) check if x/y coordinates are correct and not swapped
    cv::Mat_<double> depth_map = dataset->readDepthImage(1, 0);
    EXPECT_FALSE(depth_map.empty());
    EXPECT_DOUBLE_EQ(depth_map(0, 0), 3.04688);
    EXPECT_DOUBLE_EQ(depth_map(0, 1), 3.04688);
    EXPECT_DOUBLE_EQ(depth_map(1, 0), 3.04492);
    EXPECT_DOUBLE_EQ(depth_map(479, 639), 1.88086);
}

TEST_F(DatasetTests, readBlurredImage) {
    cv::Mat_<uchar> blurred_img = dataset->readBlurredImage(1, 0);
    EXPECT_EQ(1, blurred_img.channels());
    EXPECT_EQ(CV_8UC1, blurred_img.depth());
    EXPECT_FALSE(blurred_img.empty());
}

TEST_F(DatasetTests, readSharpImage) {
    cv::Mat_<uchar> sharp_img = dataset->readSharpImage(1, 0);
    EXPECT_EQ(1, sharp_img.channels());
    EXPECT_EQ(CV_8UC1, sharp_img.depth());
    EXPECT_FALSE(sharp_img.empty());
}

TEST_F(DatasetTests, getPose) {
    // pose of cam0 in frame 1
    mrpt::poses::CPose3DQuat pose0 = dataset->getPose(1, 0);
    EXPECT_DOUBLE_EQ(-0.684809, pose0.m_coords[0]);
    EXPECT_DOUBLE_EQ(1.59021, pose0.m_coords[1]);
    EXPECT_DOUBLE_EQ(0.91045, pose0.m_coords[2]);

    // pose of cam1 in frame 1
    // values calculated with matlab
    // 1. convert quaternions to rotation matrix R_IB
    // 2. B_P  = [0, -0.4, 0]'
    // 3. I_B  = [-0.684809, 1.59021, 0.91045]'
    // 4. T_IB = [R_IB I_B; 0 0 0 1]
    // 5. T_IB * [B_P 1]'
    mrpt::poses::CPose3DQuat pose1 = dataset->getPose(1, 1);
    EXPECT_NEAR(-0.9437, pose1.m_coords[0], 1e-4);
    EXPECT_NEAR(1.2853, pose1.m_coords[1], 1e-4);
    EXPECT_NEAR(0.9097, pose1.m_coords[2], 1e-4);

    // check rotation
    // the camera rotation quaternion maps the axis in the following form
    // camera   -> vehicle axis
    // x        -> -y
    // y        -> -z
    // z        -> x
    // this means that the camera principal axis (z) point in the x direction of the vehicle coordinate frame
    // calculate the values like this:
    // 1. R_BP = [0 0 1; -1 0 0; 0 -1 0];
    // 2. R_IP = R_IB * R_BP
    // 3. convert R_IP to quaternions
    // the orientation should be the same for both cameras
    EXPECT_NEAR(0.2731782, pose0.m_quat.r(), 5e-5);
    EXPECT_NEAR(-0.319106, pose0.m_quat.x(), 5e-5);
    EXPECT_NEAR(0.6907269, pose0.m_quat.y(), 5e-5);
    EXPECT_NEAR(-0.5885927, pose0.m_quat.z(), 5e-5);

    EXPECT_NEAR(0.2731782, pose1.m_quat.r(), 5e-5);
    EXPECT_NEAR(-0.319106, pose1.m_quat.x(), 5e-5);
    EXPECT_NEAR(0.6907269, pose1.m_quat.y(), 5e-5);
    EXPECT_NEAR(-0.5885927, pose1.m_quat.z(), 5e-5);
}
