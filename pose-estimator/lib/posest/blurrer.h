#ifndef LIB_POSEST_BLURRER_H_
#define LIB_POSEST_BLURRER_H_

#include <mrpt/poses.h>
#include "posest/posest.h"

namespace posest {
/**
 * An implementation of the Blurrer interface.
 * The class can generate artificially motion blurred images using a #Reprojector
 */
class BlurrerImpl : public Blurrer {
    const mrpt::poses::CPose3DQuat &ref_pose;
    const double ref_time;
    const double end_time;
    const double exposure_time;
    const int n_images;
    const cv::Mat &ref_image;
    const Reprojector &reprojector;
    const double image_scale;

 public:
    /**
     *
     * @param ref_pose camera pose of reference image given in world coordinates
     * @param ref_time timestamp of reference image
     * @param end_time timestamp of blurred image (when shutter closes)
     * @param exposure_time exposure time of the motion blurring camera in seconds
     * @param n_images number of images to average
     * @param ref_image sharp reference image
     * @param reprojector instance of a reprojector
     */
    BlurrerImpl(const mrpt::poses::CPose3DQuat &ref_pose,
                const double ref_time,
                const double end_time,
                const double exposure_time,
                const int n_images,
                const cv::Mat &ref_image,
                const Reprojector &reprojector,
                const double image_scale = 1);

    /**
     * Generate a blurred image and store it in blurred_img
     * @param end_pose camera pose given in world coordinates at which to genereate the blurred image
     * @param blurred_img
     */
    void blur(const mrpt::poses::CPose3DQuat &end_pose,
              cv::Mat_<uchar> &blurred_img) const override;

    /**
     * Generate a blurred image and store it in blurred_img
     * Furthermore generate store a binary mask in blurred_img_mask.
     *
     * @param end_pose
     * @param blurred_img camera pose given in world coordinates at which to genereate the blurred image
     * @param blurred_img_mask if true there is information for the corresponding pixel, false otherwise
     */
    void blur(const mrpt::poses::CPose3DQuat &end_pose,
              cv::Mat_<uchar> &blurred_img, cv::Mat_<bool> &blurred_img_mask) const override;

 private:
    /**
     * Compute interpolated pose at timestep tau_m (cf. eqn. (6) on arXiv: 1709.05745v1), with terminal pose
     * end_pose and terminal time end_time.
     *
     * Arguments:
     * - const double tau_m: time step at which to compute the new pose;
     * - const mrpt::math::CArrayDouble<6> delta_P_ts: logarithmic map delta_P_t,s that represents the difference
     *                                                 pose between the terminal and the starting poses;
     * - const Pose &end_pose: pose at terminal point of the trajectory;
     *
     * Output:
     * - CPose3D interpolatedPose: pose computed by interpolation at timestep tau_m.
     *
     */
    mrpt::poses::CPose3D interpolatedPose(const double tau_m, const mrpt::math::CArrayDouble<6> &delta_P_ts,
                                          const mrpt::poses::CPose3DQuat &end_pose) const;
};
};  // namespace posest

#endif  // LIB_POSEST_BLURRER_H_
