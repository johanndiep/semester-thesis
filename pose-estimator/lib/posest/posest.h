/**
 * This file contains interface definitions, which are used between different parts.
 * It also contains public definition files
 */

#ifndef LIB_POSEST_POSEST_H_
#define LIB_POSEST_POSEST_H_

#include <mrpt/poses/CPose3DQuat.h>
#include <opencv2/core/mat.hpp>
#include <Eigen/Dense>
#include <ostream>

namespace posest {

/**
 * Internal calibration of a camera
 * Basically the K matrix
 */
struct InternalCalibration {
    double fx;
    double fy;
    double cx;
    double cy;

    /**
     * calculate the 3x3 K Matrix
     * @param dst resulting 3x3 matrix
     */
    inline const Eigen::Matrix3d toMatrix() const {
        Eigen::Matrix3d m;
        m << fx, 0, cx, 0, fy, cy, 0, 0, 1;
        return m;
    }

    friend std::ostream &operator<<(std::ostream &os, const InternalCalibration &calibration) {
        os << "fx: " << calibration.fx << " fy: " << calibration.fy
           << " cx: " << calibration.cx << " cy: " << calibration.cy;
        return os;
    }
};

/**
 * Reprojector interface
 * A reprojector can warp a sharp reference image together with its depth map to a camera at a different pose.
 */
class Reprojector {
 public:
    /**
     * Reprojects a ref image to a second camera at a slightly different pose
     *
     * @param ref_img sharp reference image with depth map depth taken at pose ref
     * @param depth depth map for each pixel given as double matrix
     * @param ref_pose pose of reference frame
     * @param k internal camera calibration
     * @param reproj_pose pose of the target camera where the image is reprojected to
     * @param reproj_img reprojected image (output of the function)
     */
    virtual void reproject(const mrpt::poses::CPose3DQuat &reproj_pose, cv::Mat_<uchar> &reproj_img) const = 0;

    /**
     * Reprojects the 3D point cloud to position reproj_pose
     * @param reproj_pose target position for warping
     * @param reproj_img reprojected image
     * @param mask boolean Mat which is true if the value of reproj_img at the same point is valid
     */
    virtual void reproject(const mrpt::poses::CPose3DQuat &reproj_pose,
                           cv::Mat_<uchar> &reproj_img,
                           cv::Mat_<bool> &mask) const = 0;
};

/**
 * A Blurrer can generate artificially blurred images.
 */
class Blurrer {
 public:
    /**
     * Generate a motion blurred image.
     * @param end_pose end pose towards which the image should be motion blurred
     * @param blurred_img where to write the blurred image
    */
    virtual void blur(const mrpt::poses::CPose3DQuat &end_pose,
                      cv::Mat_<uchar> &blurred_img) const = 0;

    /**
     * Generate a motion blurred image
     * @param end_pose end pose towards which the image should be motion blurred
     * @param blurred_img where to write the blurred image
     * @param blurred_img_mask where to store a binary mask with all valid pixels set to true
     */
    virtual void blur(const mrpt::poses::CPose3DQuat &end_pose,
                      cv::Mat_<uchar> &blurred_img, cv::Mat_<bool> &blurred_img_mask) const = 0;
};

}  // namespace posest

#endif  // LIB_POSEST_POSEST_H_
