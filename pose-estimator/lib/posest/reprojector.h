#ifndef LIB_POSEST_REPROJECTOR_H_
#define LIB_POSEST_REPROJECTOR_H_

#include "posest/posest.h"
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <fstream>
#include <math.h>
#include <vector>


// #define SerialPrintLoopCouts

namespace posest {

/**
 * ReprojectorImpl is a Reprojector implementation.
 * It can warp a sharp reference image together with its depth map into a camera at an other pose.
 */
class ReprojectorImpl : public Reprojector {
    std::vector<mrpt::poses::CPoint3D> points3D;
    const InternalCalibration internal_calibration;
    const cv::Mat_<uchar> ref_img;
    const double image_scale;

 public:
    /**
     *
     * @param ic
     * @param ref_img
     * @param ref_depth
     * @param ref_pose
     */
    ReprojectorImpl(const posest::InternalCalibration &ic,
                    const cv::Mat_<uchar> &ref_img,
                    const cv::Mat_<double> &ref_depth,
                    const mrpt::poses::CPose3DQuat &ref_pose,
                    const double image_scale = 1);

    /**
     *
     * @return
     */
    const std::vector<mrpt::poses::CPoint3D> &getPoints3D() const { return points3D; }

    /**
     * Reprojects the 3D point cloud into a camera at pose reproj_pose.
     * The resulting vector pixel_coords has the same order like the points3D vector.
     * @param reproj_pose the camera pose to project the the 3D-point cloud into
     * @param pixel_coords the resulting pixel coordinates are written into this vector. The z axis is the distance from
     *                      the cam.
     */
    void reproject(const mrpt::poses::CPose3DQuat &reproj_pose, cv::Mat_<uchar> &reproj_img) const override;

    /**
     *
     * @param reproj_pose
     * @param pixel_coords
     */
    void reproject(const mrpt::poses::CPose3DQuat &reproj_pose, std::vector<mrpt::math::TPoint3D> &pixel_coords, const double image_scale = 1) const;

    /**
     *
     * @param reproj_pose
     * @param reproj_img
     * @param mask
     */
    void reproject(const mrpt::poses::CPose3DQuat &reproj_pose,
                   cv::Mat_<uchar> &reproj_img,
                   cv::Mat_<bool> &mask) const override;
};

};  // namespace posest


#endif  // LIB_POSEST_REPROJECTOR_H_
