#include "posest/blurrer.h"
#include <vector>
#include <opencv2/highgui.hpp>
#include <cassert>
#include "mrpt/math.h"

using mrpt::poses::CPose3D;
using mrpt::poses::CPose3DQuat;
using mrpt::math::CArrayDouble;

namespace posest {
BlurrerImpl::BlurrerImpl(const CPose3DQuat &ref_pose, const double ref_time, const double end_time,
                         const double exposure_time, const int n_images, const cv::Mat &ref_image,
                         const Reprojector &reprojector)
        : ref_pose(ref_pose), ref_time(ref_time), end_time(end_time), exposure_time(exposure_time), n_images(n_images),
          ref_image(ref_image), reprojector(reprojector) {}

void BlurrerImpl::blur(const CPose3DQuat &end_pose, cv::Mat_<uchar> &blurred_img,
                       cv::Mat_<bool> &blurred_img_mask) const {
    // Check that exposure time is consistent with start time (i.e., time corresponding to the reference pose) and
    // end time
    assert(exposure_time + ref_time <= end_time);

    // Convert poses to CPose3DAdded reprojector and ref_image to Blurrer constructor
    CPose3D P_s(ref_pose);

    CPose3D P_t(end_pose);

    // Compute delta_P_ts (cf. eqn. (6) on arXiv: 1709.05745v1)
    CPose3D pose_composition;
    pose_composition.inverseComposeFrom(P_t, P_s);
    CArrayDouble<6> delta_P_ts = pose_composition.ln();

    // Compute new poses at each intermediate point tau_m
    std::vector<double> timesteps;
    mrpt::math::linspace(end_time - exposure_time, end_time, n_images, timesteps);

    cv::Mat_<int> sum_of_images = cv::Mat_<int>::zeros(ref_image.size());
    cv::Mat_<int> sum_of_masks = cv::Mat_<int>::zeros(ref_image.size());
    blurred_img_mask = cv::Mat_<bool>::zeros(ref_image.size());
    cv::Mat_<uchar> curr_image(ref_image.size());
    cv::Mat_<bool> curr_mask(ref_image.size());

    for (int step_index = 0; step_index < n_images; step_index++) {
        // Compute pose at current timestep
        CPose3D curr_pose = interpolatedPose(timesteps[step_index], delta_P_ts, end_pose);
        // Note: to handle overflows, it is necessary to use CV_64F or int images, further references: https://goo.gl/5yLyCn

        // clear previous results
        curr_image.setTo(0);
        curr_mask.setTo(false);

        // Obtain projected image according to pose
        reprojector.reproject(CPose3DQuat(curr_pose), curr_image, curr_mask);

        sum_of_images += curr_image;
        sum_of_masks += curr_mask;
    }
    auto iter = blurred_img_mask.begin();
    for (auto p : sum_of_masks) {
        if (p > n_images / 5) {
            *iter = true;
        }
        iter++;
    }
    // Assign averaged image to output image
    sum_of_images /= sum_of_masks;
    sum_of_images.convertTo(blurred_img, CV_8U);
}

CPose3D BlurrerImpl::interpolatedPose(const double tau_m, const CArrayDouble<6> &delta_P_ts,
                                      const CPose3DQuat &end_pose) const {
    CPose3D P_tau_m;

    // Scales difference pose
    CArrayDouble<6> delta_P_ts_scaled = delta_P_ts;
    for (int i = 0; i < delta_P_ts_scaled.size(); i++)
        delta_P_ts_scaled[i] *= (tau_m - ref_time) / (end_time - ref_time);

    // Compute exponential map
    CPose3D exp_map_pose = CPose3D::exp(delta_P_ts_scaled);

    // Compose map resulting from exponential map and starting pose
    P_tau_m.composeFrom(CPose3D(ref_pose), exp_map_pose);

    return P_tau_m;
}

void BlurrerImpl::blur(const mrpt::poses::CPose3DQuat &end_pose, cv::Mat_<uchar> &blurred_img) const {
    cv::Mat_<bool> mask;
    blur(end_pose, blurred_img, mask);
}

};  // namespace posest
