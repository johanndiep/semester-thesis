/**
 * coarse-to-sparse tries to find the position where a blurred image was taken from.
 * The steps are equal as in pose-estimation, with the difference that it uses pyramid scale resolution
 * and uses previous results as initiliation for the current iteration. After a number of pyramid steps and convergence,
 * it calculates the remaining error and writes the result to a file.
 */

#include <iostream>
#include <posest/dataset.h>
#include <random>
#include "posest/executionparametrization.h"
#include <mrpt/poses.h>
#include <mrpt/math.h>
#include <cmath>
#include <string>

using posest::Dataset;
using std::cerr;
using std::endl;
using mrpt::poses::CPoint3D;
using mrpt::math::CQuaternionDouble;
using mrpt::math::CArrayDouble;
using std::cout;
using std::endl;

/**
 * Reproject the image without blurring, but still implement the Blurrer interface
 */
class NoOpBlurrer : public posest::Blurrer {
    const posest::Reprojector &reprojector;
 public:
    explicit NoOpBlurrer(const posest::Reprojector &reprojector) : reprojector(reprojector) {}

    void blur(const CPose3DQuat &end_pose, Mat_<uchar> &blurred_img) const override {
        reprojector.reproject(end_pose, blurred_img);
    }
};

/**
 * Randomizer can generate random vectors and orientations
 */
class Randomizer {
    std::default_random_engine generator;
    std::uniform_real_distribution<double> theta_distribution;
    std::uniform_real_distribution<double> phi_distribution;

 public:
    Randomizer() : generator(static_cast<unsigned int>(time(nullptr))), theta_distribution(0, M_PI),
                   phi_distribution(0, M_2_PI) {}

    /**
     * Sample a vector with random orientation and given length
     * The idea is to sample angles of sphere coordinates and then create a vector of given length
     * @param dist
     */
    CPose3DQuat rand_position_offset(double dist) {
        double phi = phi_distribution(generator);
        double theta = theta_distribution(generator);
        double x = dist * std::sin(theta) * cos(phi);
        double y = dist * std::sin(theta) * sin(phi);
        double z = dist * std::cos(theta);
        return CPose3DQuat(x, y, z, CQuaternionDouble());
    }

    /**
     * rand_pose_offset creates new CPose3DQuat.
     * @param dist length of the vector
     * @param angle angle in radians
     * @return
     */
    CPose3DQuat rand_pose_offset(double dist, double angle) {
        CPose3DQuat position = rand_position_offset(dist);
        CPose3DQuat angular_vec = rand_position_offset(angle);
        CQuaternionDouble q;
        q.fromRodriguesVector(angular_vec.m_coords);
        return CPose3DQuat(position.m_coords.x(),
                           position.m_coords.y(),
                           position.m_coords.z(),
                           q);
    }
};

int main(int argc, char *argv[]) {
	// minimal arguments check
	if (argc < 11) {
		std::cerr << "coarse-to-sparse tries to find the position where a blurred image was taken from." << std::endl;
		std::cerr << "The steps are equal as in pose-estimation, with the difference that it uses pyramid scale resolution" << std::endl;
		std::cerr << "and uses previous results as initiliation for the current iteration. After a number of pyramid steps" << std::endl;
		std::cerr << "and convergence, it calculates the remaining error and writes the result to a file." << std::endl;
		std::cerr << std::endl;
		std::cerr << "Usage: " << argv[0]
				  << " [dataset_path] [cam_index] [ref_img_index] [blurred_img_index] [n_images] [initial_offset_pos] "
                     "[initial_offset_rot] [sigma] [output_file] [pyramid height]"
                  << endl;
		return 1;
	}

	// parse command line arguments
	posest::ExecutionParametrization params;
	const std::string dataset_path(argv[1]);
	params.cam_index = atoi(argv[2]);
	params.ref_img_index = atoi(argv[3]);
    params.blurred_img_index = atoi(argv[4]);
    params.n_images = atoi(argv[5]);
    const double initial_offset_pos = atof(argv[6]);
    const double initial_offset_rot = atof(argv[7]);
    params.sigma = atof(argv[8]);
    const std::string output_file(argv[9]);
    const int pyramid_height = atoi(argv[10]);

    // print out parameterization
}