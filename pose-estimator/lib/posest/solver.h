#ifndef LIB_POSEST_SOLVER_H_
#define LIB_POSEST_SOLVER_H_

#include "posest/posest.h"
#include <opencv2/core/mat.hpp>
#include "ceres/ceres.h"
#include <string>

namespace posest {
/**
 * A Solver sets up an optimization problem, which is then solved for the the pose of the blurred input image.
 * It uses a #Blurrer to generate artificially blurred images to compare the blurred input with.
 */
class Solver {
    const cv::Mat_<uchar> &blurred_input;
    const Blurrer &blurrer;
    mrpt::poses::CPose3DQuat result;
    std::string snapshot_path;

 public:
    /**
     * Initialize a new solver
     * @param blurred_input the image for which the camere pose should be found
     * @param interpolator an instance of a blurrer
     */
    Solver(const cv::Mat_<uchar> &blurred_input, const Blurrer &interpolator) : blurred_input(blurred_input),
                                                                                blurrer(interpolator) {}

    /**
     * Solve starts the solving process
     * @param initial_guess pose to initialize the solver
     * @return the pose after convergence
     */
    const mrpt::poses::CPose3DQuat &solve(mrpt::poses::CPose3DQuat &initial_guess);

    /**
     * Solve starts the solving process
     * @param initial_guess pose to initialize the solver
     * @param summary a summary is written to this variable
     * @return the pose after convergence
     */
    const mrpt::poses::CPose3DQuat &solve(mrpt::poses::CPose3DQuat &initial_guess, ceres::Solver::Summary &summary);

    /**
     * If a snapshot path is set. The solver saves a blurred image after each iteration.
     * @param path where to save the image
     */
    void set_snapshot_path(const std::string &path) {
        snapshot_path = std::string(path);
    }
};

};  // namespace posest

#endif  // LIB_POSEST_SOLVER_H_
