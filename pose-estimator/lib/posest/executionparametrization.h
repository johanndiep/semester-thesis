#ifndef LIB_POSEST_EXECUTIONPARAMETRIZATION_H_
#define LIB_POSEST_EXECUTIONPARAMETRIZATION_H_

#include "posest/posest.h"
#include <posest/dataset.h>
#include <posest/solver.h>
#include "posest/blurrer.h"
#include "posest/reprojector.h"
#include <string>

using cv::Mat_;
using mrpt::poses::CPose3DQuat;
using mrpt::poses::CPose3DRotVec;

namespace posest {
/**
* Container for results of our pose estimator
*/
class ExecutionResults {
    // ground truth camera pose
    const CPose3DQuat exact_pose;
    // pose error between solved pose and exact pose
    const CPose3DQuat pose_error;
    // pose where the solver converged to
    const CPose3DQuat solved_pose;
    // summary report of the ceres solver
    const ceres::Solver::Summary summary;

 public:
    ExecutionResults() = default;

    /**
     * Initialize a Execution results.
     * @param exact_pose ground truth pose
     * @param solved_pose pose where the solver converged to
     * @param summary ceres summary report
     */
    ExecutionResults(const CPose3DQuat &exact_pose, const CPose3DQuat &solved_pose,
                     const ceres::Solver::Summary &summary) :
            exact_pose(exact_pose), pose_error(exact_pose - solved_pose), solved_pose(solved_pose), summary(summary) {}

    /**
     * Returns the position error between ground truth orientation and the solved orientation.
     * (coordinates are given in meters)
     * @return
     */
    const mrpt::math::CArrayDouble<3> &get_position_error() const {
        return pose_error.m_coords;
    }

    /**
     * Returns the error between ground truth orientation and the solved orientation as a quaternion
     * @return
     */
    const mrpt::math::CQuaternionDouble &get_rotation_error() const {
        return pose_error.m_quat;
    }

    /**
     * Returns the distance between ground truth camera position and the solved position in meters
     * @return
     */
    double get_distance_error() const {
        return get_position_error().norm();
    }

    /**
     * Returns the minimal angle between ground truth orientation and the solved orientation in radians.
     * @return
     */
    double get_angular_error() const {
        mrpt::poses::CPose3D p(pose_error);
        CPose3DRotVec err(p);
        std::cout << err << std::endl;
        return err.m_rotvec.norm();
    }

    /**
     * Returns the number of iterations it took the solver to converge
     * @return
     */
    int get_num_iterations() const {
        return static_cast<int>(summary.iterations.size());
    }

    /**
     * Returns the total time the solver used to solve the problem
     * @return
     */
    double get_total_time() const {
        return summary.total_time_in_seconds;
    }

    /**
     * Returns true iff the solver converged
     * @return
     */
    bool has_converged() const {
        return summary.termination_type == ceres::CONVERGENCE;
    }
};

/**
 * ExecutionParametrization the struct which needs to be initialized at the very first
 * before one can use the posest. It takes the initial ref image and the blurred image
 * and the amount of images for the shutter (to create the artifical blurred img)
 */
class ExecutionParametrization {
    ExecutionResults *execution_results;

 public:
    // camera index (usually we use cam0)
    int cam_index;
    // index of the sharp reference image
    int ref_img_index;
    // index of the blurred input image whose pose should be solved for
    int blurred_img_index;
    // number of images used in the blurring step
    int n_images;
    // offset applied to the exact pose of the blurred input image
    CPose3DQuat initial_offset;
    // path where snapshots after each solver iteration should be stored
    std::string snapshot_path;
    bool snap_only_final;
    // amount of blurring applied to depth map of sharp reference frame
    double sigma;

    ExecutionParametrization() : execution_results(nullptr), cam_index(0), ref_img_index(1), blurred_img_index(2),
                                 n_images(5), snapshot_path(), snap_only_final(true), sigma(0) {}  // default

    /**
     * Execute the whole algorithm for the dataset
     * @param dataset
     * @return
     */
    const ExecutionResults *posest_start(const Dataset &dataset) {
        if (execution_results != nullptr) delete (execution_results);
        // read information from dataset
        const double image_scale = 4; // set this parameter to 1 if ctp should be deactivated
        //const Mat_<uchar> &ref_sharp = dataset.readSharpImage(ref_img_index, cam_index);
        const Mat_<uchar> &ref_sharp = dataset.readSharpScaledImage(ref_img_index, cam_index, image_scale);
        const CPose3DQuat &ref_pose = dataset.getPose(ref_img_index, cam_index);
        //const Mat_<uchar> &blurred_img = dataset.readBlurredImage(blurred_img_index, cam_index);
        const Mat_<uchar> &blurred_img = dataset.readBlurredScaledImage(blurred_img_index, cam_index, image_scale);
        const InternalCalibration &internalCalibration = dataset.getInternalCalibration(cam_index);
        const double exposure_time = dataset.getExposureTime();
        const CPose3DQuat &blurred_exact_pose = dataset.getPoseAtTime(blurred_img_index * 0.1, cam_index);
        Mat_<double> ref_depth;
        if (sigma == 0) {
            ref_depth = dataset.readDepthImage(ref_img_index, cam_index);
            ref_depth = dataset.readScaledDepthImage(ref_depth, image_scale);
        } 
        else {
            ref_depth = dataset.readDepthImage(ref_img_index, cam_index, sigma);
            ref_depth = dataset.readScaledDepthImage(ref_depth, image_scale);

        }

        // calculate pose where solver should start in the first iteration
        CPose3DQuat initial_pose = blurred_exact_pose + initial_offset;

        // setup pipeline with reprojector, blurrer and solver
        posest::ReprojectorImpl reprojector(internalCalibration, ref_sharp, ref_depth, ref_pose, image_scale);
        posest::BlurrerImpl blurrer(ref_pose,
                                    ref_img_index * 0.1,
                                    blurred_img_index * 0.1,
                                    dataset.getExposureTime(),
                                    n_images,
                                    ref_sharp,
                                    reprojector,
                                    image_scale);
        posest::Solver solver(blurred_img, blurrer);
        if (!snap_only_final && snapshot_path.length()) {
            solver.set_snapshot_path(snapshot_path);
        }

        // start the pipeline
        ceres::Solver::Summary summary;
        const CPose3DQuat &solved_pose = solver.solve(initial_pose, summary);

        // prepare results to return
        execution_results = new ExecutionResults(
                blurred_exact_pose,
                solved_pose,
                summary);
        if (snap_only_final && snapshot_path.length()) {
            Mat_<uchar> snap = Mat_<uchar>::zeros(blurred_img.size());
            blurrer.blur(solved_pose, snap);
            cv::imwrite(snapshot_path, snap);
        }
        return execution_results;
    }

    virtual ~ExecutionParametrization() {
        if (execution_results != nullptr)
            delete (execution_results);
    }
};
};  // namespace posest

#endif  // LIB_POSEST_EXECUTIONPARAMETRIZATION_H_
