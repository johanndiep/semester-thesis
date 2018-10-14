#include <mrpt/poses/CPose3DQuat.h>
#include "posest/reprojector.h"
#include <limits>
#include <cmath>

// TODO(cschnetz) avoid global variables
int MaxRows;
int MaxCols;
bool IsContinuous;

using mrpt::poses::CPose3DQuat;
using mrpt::math::TPoint3D;
using cv::Mat_;


namespace posest {

ReprojectorImpl::ReprojectorImpl(const posest::InternalCalibration &ic,
                                 const cv::Mat_<uchar> &ref_img,
                                 const cv::Mat_<double> &ref_depth,
                                 const mrpt::poses::CPose3DQuat &ref_pose,
                                 const double image_scale)
        : internal_calibration(ic), ref_img(ref_img) {
    // allocate enough space for each pixel
    points3D.reserve(ref_img.total());

    // calculate corresponding 3D point from reference image using the internal calibration and ref_depth.
    // The 3D points are stored in points3D in row major order. Use this order to get the original pixel color value
    // from the original
    for (int pix_y = 0; pix_y < ref_img.rows; pix_y++) {
        for (int pix_x = 0; pix_x < ref_img.cols; pix_x++) {
            // coordinates in camera frame
            double z_file = ref_depth(pix_y, pix_x);  // in meters (the hypothenuse)
            double cam_z = z_file / std::sqrt(
                    ((static_cast<double>(pix_x) - (ic.cx/image_scale)) / ic.fx/image_scale) * ((static_cast<double>(pix_x) - (ic.cx/image_scale)) / ic.fx/image_scale) +
                    ((static_cast<double>(pix_y) - (ic.cy/image_scale)) / ic.fy/image_scale) * ((static_cast<double>(pix_y) - (ic.cy/image_scale)) / ic.fy/image_scale)
                    + 1);
            double cam_x = cam_z * (static_cast<double>(pix_x) - (ic.cx/image_scale)) / ic.fx/image_scale;
            double cam_y = cam_z * (static_cast<double>(pix_y) - (ic.cy/image_scale)) / ic.fy/image_scale;
            const mrpt::poses::CPoint3D cam_p(cam_x, cam_y, cam_z);

            // calculate coordinates in world frame and store in array
            points3D.push_back(ref_pose + cam_p);
        }
    }
}

void ReprojectorImpl::reproject(const CPose3DQuat &reproj_pose, Mat_<uchar> &reproj_img, Mat_<bool> &mask, const double image_scale) const {
    assert(reproj_img.channels() == 1);
    reproj_img.create(ref_img.size());
    mask.create(ref_img.size());

    std::vector<mrpt::math::TPoint3D> pixel_coords;
    reproject(reproj_pose, pixel_coords, image_scale);

    cv::Rect img_bounds(cv::Point(), reproj_img.size());
    auto ref_img_itr = ref_img.begin();
    cv::Mat_<double> pixel_depth(reproj_img.size(), std::numeric_limits<double>::infinity());

    for (auto &pix_p : pixel_coords) {
        int pix_x = static_cast<int>(std::round(pix_p.x));
        int pix_y = static_cast<int>(std::round(pix_p.y));

        // only use the pixel if it is closer than any other pixel that maps to this pixel that has already been
        // processed. Skip pixels with a negative z axis as they are behind the camera and are not captured.
        if (img_bounds.contains(cv::Point(pix_x, pix_y))
            && pix_p.z < pixel_depth.at<double>(pix_y, pix_x) && pix_p.z > 0) {
            reproj_img.at<uchar>(pix_y, pix_x) = *ref_img_itr;
            pixel_depth.at<double>(pix_y, pix_x) = pix_p.z;
            mask.at<bool>(pix_y, pix_x) = true;
        }

        ref_img_itr++;
    }
}

/**
 * Reprojects the 3D point cloud into a camera at pose reproj_pose.
 * The resulting vector pixel_coords has the same order like the points3D vector.
 * @param reproj_pose the camera pose to project the the 3D-point cloud into
 * @param pixel_coords the resulting pixel coordinates are written into this vector. The z axis is the distance from
 *                      the cam.
 */
void ReprojectorImpl::reproject(const CPose3DQuat &reproj_pose, std::vector<TPoint3D> &pixel_coords, const double image_scale) const {
    // allocate enough space for each pixel
    pixel_coords.clear();
    pixel_coords.reserve(points3D.size());

    // loop over each 3D point and project it into the camera at reproj_pose.
    // store the resulting 2D pixel in pixel_coords.
    for (const auto &world_p : points3D) {
        // calculate p in reproj camera pose frame
        mrpt::poses::CPoint3D cam_p = world_p - reproj_pose;

        const auto pix_x =
                static_cast<float>(cam_p.x() * internal_calibration.fx / (cam_p.z() * image_scale) + (internal_calibration.cx/image_scale));
        const auto pix_y =
                static_cast<float>(cam_p.y() * internal_calibration.fy / (cam_p.z() * image_scale) + (internal_calibration.cy/image_scale));

        pixel_coords.emplace_back(pix_x, pix_y, cam_p.z());
    }
}

void ReprojectorImpl::reproject(const mrpt::poses::CPose3DQuat &reproj_pose, cv::Mat_<uchar> &reproj_img) const {
    cv::Mat_<bool> mask;
    reproject(reproj_pose, reproj_img, mask);
}
}  // namespace posest
