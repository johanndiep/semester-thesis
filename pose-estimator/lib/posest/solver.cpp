#include "posest/solver.h"
#include <sstream>
#include <iomanip>
#include "ceres/rotation.h"
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <sys/stat.h>
#include <libgen.h>

using ceres::CostFunction;
using ceres::Problem;
using ceres::Solve;
using ceres::NumericDiffCostFunction;
using cv::Mat_;
using mrpt::poses::CPose3DQuat;

namespace posest {

/**
 * Image Residual calculates the L2 norm between two images.
 * Empty pixels are penalized with a constant.
 * It is used by ceres-solver to minimize the residuals by changing the
 * camera parametrization.
 *
 * btw. ceres-solver uses hamiltonian quaternion convention
 */
class ImageResidual {
    const Mat_<double> input_image;
    const Blurrer &blurrer;

 public:
    ImageResidual(const Mat_<uchar> &input, const Blurrer &blurrer) :
            input_image(input.size()), blurrer(blurrer) {
        input.convertTo(input_image, CV_64FC1);
    }

    bool operator()(const double *const xyz, const double *const xyzw, double *residual) const {
        // generate artificially blurred image at position parametrized by x
        double w = xyzw[0];
        double x = xyzw[1];
        double y = xyzw[2];
        double z = xyzw[3];
        double norm = std::sqrt(w * w + x * x + y * y + z * z);
        w /= norm;
        x /= norm;
        y /= norm;
        z /= norm;
        const mrpt::math::CQuaternion<double> &q = mrpt::math::CQuaternionDouble(w, x, y, z);
        CPose3DQuat pose(xyz[0], xyz[1], xyz[2], q);


        Mat_<uchar> generated_uchar(input_image.size(), 0);
        Mat_<double> generated_double(input_image.size());
        Mat_<bool> generated_mask(input_image.size(), false);

        blurrer.blur(pose, generated_uchar, generated_mask);

        generated_uchar.convertTo(generated_double, CV_64FC1);

        // calculate the residual image as
        generated_double -= input_image;

        auto mask_iter = generated_mask.begin();
        for (double &img_itr : generated_double) {
            if (*(mask_iter++)) {
                *(residual++) = img_itr;
            } else {
                *(residual++) = 10;
            }
        }

        return true;
    }
};

/**
 * SaveSnapCallback saves an image snapshot after each solving iteration
 */
class SaveSnapCallback : public ceres::IterationCallback {
    const mrpt::poses::CPose3DQuat &pose;
    const posest::Blurrer &blurrer;
    const std::string path;
    std::ofstream pose_log;

 public:
    SaveSnapCallback(const mrpt::poses::CPose3DQuat &pose, const Blurrer &blurrer, std::string &path) :
            pose(pose), blurrer(blurrer), path(path), pose_log() {
        mkdir(path.c_str(), S_IRWXU);
        pose_log.open(path + std::string("/pose-iteration-log.csv"));
        pose_log << "x,y,z" << std::endl;
    }

    ceres::CallbackReturnType operator()(const ceres::IterationSummary &summary) override {
        if (summary.cost_change < 0) {
            return ceres::SOLVER_CONTINUE;
        }
        // log intermediate pose
        pose_log << pose.x() << "," << pose.y() << "," << pose.z() << std::endl;

        // write snapshot
        cv::Mat_<uchar> img;
        blurrer.blur(pose, img);
        std::stringstream ss;
        ss << path << "/";
        ss << "snapshot-";
        ss << std::setfill('0') << std::setw(3) << summary.iteration;
        ss << ".png";
        cv::imwrite(ss.str(), img);
        return ceres::SOLVER_CONTINUE;
    }

    virtual ~SaveSnapCallback() {
        this->pose_log.close();
    }
};

class NumericDiffStepSizeAdjuster : public ceres::IterationCallback {
    ceres::NumericDiffOptions &options;
 public:
    explicit NumericDiffStepSizeAdjuster(ceres::NumericDiffOptions &options) : options(options) {}

    ceres::CallbackReturnType operator()(const ceres::IterationSummary &summary) override {
        if (summary.cost_change < 0) {
            return ceres::SOLVER_CONTINUE;
        }
        options.relative_step_size *= 0.90;
        return ceres::SOLVER_CONTINUE;
    }
};

const mrpt::poses::CPose3DQuat &Solver::solve(CPose3DQuat &initial_guess) {
    ceres::Solver::Summary s;
    return solve(initial_guess, s);
}

const mrpt::poses::CPose3DQuat &Solver::solve(CPose3DQuat &initial_guess, ceres::Solver::Summary &summary) {
    ceres::NumericDiffOptions diff_opts;
    diff_opts.relative_step_size = 0.05;
    // diff_opts.ridders_relative_initial_step_size= 0.05 ;
    // diff_opts.max_num_ridders_extrapolations = 10;

    CostFunction *cost_function = new NumericDiffCostFunction<ImageResidual, ceres::CENTRAL, ceres::DYNAMIC, 3, 4>(
            new ImageResidual(blurred_input, blurrer),
            ceres::TAKE_OWNERSHIP,
            static_cast<int>(blurred_input.total()),
            diff_opts);
    double *xyz = initial_guess.xyz().data();
    double *xyzw = initial_guess.m_quat.data();
    ceres::Problem problem;

    problem.AddResidualBlock(cost_function, nullptr, xyz, xyzw);
    problem.SetParameterization(xyzw, new ceres::QuaternionParameterization());

    ceres::Solver::Options options;
    options.max_num_iterations = 250;
    options.minimizer_progress_to_stdout = true;
    options.update_state_every_iteration = true;
    // options.function_tolerance = 1e-4;
    // options.parameter_tolerance = 2e-7;

    if (snapshot_path.length() > 0) {
        options.callbacks.push_back(new SaveSnapCallback(initial_guess, blurrer, snapshot_path));
    }
    options.callbacks.push_back(new NumericDiffStepSizeAdjuster(diff_opts));

    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << "\n";

    std::cout << "final relative step size: " << diff_opts.relative_step_size << std::endl;

    result = CPose3DQuat(xyz[0], xyz[1], xyz[2], mrpt::math::CQuaternionDouble(xyzw[0], xyzw[1], xyzw[2], xyzw[3]));
    return result;
}
};  // namespace posest
