#include <gtest/gtest.h>
#include <posest/reprojector.h>
#include <posest/dataset.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <string>

class EigenReprojectorTests : public ::testing::Test {
 protected:
    posest::ReprojectorImpl *reprojector;
    posest::Dataset *dataset;

    virtual void SetUp() {
        dataset = new posest::Dataset("/realistic-dataset");
        dataset->read();
        reprojector = new posest::ReprojectorImpl(
                dataset->getInternalCalibration(0),
                dataset->readSharpImage(2, 0),
                dataset->readDepthImage(2, 0),
                dataset->getPose(2, 0));
    }

    virtual void TearDown() {
        delete (reprojector);
        delete (dataset);
    }
};


TEST_F(EigenReprojectorTests, getPoints3D) {
    auto points3D = reprojector->getPoints3D();

    std::ofstream points3D_file;
    points3D_file.open("3d-points.txt");
    for (auto p : points3D) {
        points3D_file << p.x() << ";" << p.y() << ";" << p.z() << ";255;255;255" << std::endl;
    }
    points3D_file.close();

    std::cout << "# 3D points: " << points3D.size() << std::endl;
    std::cout << "first 3D point: " << points3D[0] << std::endl;
    std::cout << "(412, 102) 3D point: " << points3D[102 * 640 + 412] << std::endl;
}

TEST_F(EigenReprojectorTests, reproject_coords) {
    std::vector<mrpt::math::TPoint3D> pixels;
    reprojector->reproject(dataset->getPose(3, 0), pixels);

    std::cout << "reprojected pixel (  0,   0): " << pixels[0] << std::endl;
    std::cout << "reprojected pixel (100, 100): " << pixels[100 * 640 + 100] << std::endl;
    // TODO(tiaebi) do some assertions here
}

TEST_F(EigenReprojectorTests, reproject_img) {
    cv::Mat_<uchar> reproj_img = cv::Mat_<uchar>::zeros(480, 640);
    cv::Mat_<bool> mask = cv::Mat_<uchar>::zeros(480, 640);
    reprojector->reproject(dataset->getPose(5, 0), reproj_img, mask);

    //cv::Mat_<
    cv::Mat output;
    cv::cvtColor(reproj_img, output, CV_GRAY2RGB);
    for (auto &element: mask) {
        element = !element;
    }
    output.setTo(cv::Scalar(160, 250, 255), mask);
    cv::imwrite("reprojected-img.png", output);
    //cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE);     // Create a window for display.
    //cv::imshow("Display window", reproj_img);                            // Show our image inside it.

    //cv::waitKey(0);                                             // Wait for a keystroke in the window
}