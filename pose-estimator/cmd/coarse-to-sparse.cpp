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
    cout << endl << "=============================================================" << endl;
    cout << "running algorithm with: " << endl;
    cout << "         cam_index: " << params.cam_index << endl;
    cout << "      dataset_path: " << dataset_path << endl;
    cout << "     ref_img_index: " << params.ref_img_index << endl;
    cout << " blurred_img_index: " << params.blurred_img_index << endl;
    cout << "          n_images: " << params.n_images << endl;
    cout << "initial_offset_pos: " << initial_offset_pos << endl;
    cout << "initial_offset_rot: " << initial_offset_rot << endl;
    cout << "       output_file: " << output_file << endl;
    cout << "             sigma: " << params.sigma << endl;
    cout << "    pyramid_height: " << pyramid_height << endl;

    // google logging is used by ceres and needs to be initialized only once
    google::InitGoogleLogging("solver");

    // read dataset
    Dataset dataset(dataset_path);
    dataset.read();

    // add some random pose offset of defined magnitude
    Randomizer rnd;
    params.initial_offset = CPose3DQuat(rnd.rand_pose_offset(initial_offset_pos, initial_offset_rot));

    // start the solving process
    const posest::ExecutionResults *results = params.posest_start(dataset, pyramid_height);

    // get error after convergence with respect to ground truth
    const CArrayDouble<3> &err_pos = results->get_position_error();
    const CQuaternionDouble &err_rot = results->get_rotation_error();

    // get solved pose
    const CPose3DQuat &solved_pose = results->get_solved_pose();

    // write results to output file
    std::ofstream file;
    file.open(output_file + std::string(".txt"));
    file << "Layer: " << pyramid_height << endl;
    file << "cam_index: " << params.cam_index << ", ";                              // cam_index
    file << "ref_img_index: " << params.ref_img_index << ", ";                      // ref_img_index
    file << "blurred_img_index: " << params.blurred_img_index << ", ";              // blurred_img_index
    file << "sigma: "<< params.sigma << ",";                                        // depth_perturbation_sigma
    file << "initial_offset: " << params.initial_offset.m_coords[0] << ",";         // initial_offset_x
    file << params.initial_offset.m_coords[1] << ",";                               // initial_offset_y
    file << params.initial_offset.m_coords[2] << ",";                               // initial_offset_z
    file << params.initial_offset.m_quat[0] << ",";                                 // initial_offset_qw
    file << params.initial_offset.m_quat[1] << ",";                                 // initial_offset_qx
    file << params.initial_offset.m_quat[2] << ",";                                 // initial_offset_qy
    file << params.initial_offset.m_quat[3] << ", ";                                // initial_offset_qz
    file << "initial_offset_dist: " << params.initial_offset.norm() << ", ";        // initial_offset_dist
    file << "initial_offset_rot_angle: " << initial_offset_rot << ", ";             // initial_offset_rot_angle
    file << "n_images: " << params.n_images << ", ";                                // n_images
    file << "err: " << err_pos[0] << ",";                                           // err_x
    file << err_pos[1] << ",";                                                      // err_y
    file << err_pos[2] << ",";                                                      // err_z
    file << err_rot[0] << ",";                                                      // err_qw
    file << err_rot[1] << ",";                                                      // err_qx
    file << err_rot[2] << ",";                                                      // err_qy
    file << err_rot[3] << ", ";                                                     // err_qz
    file << "solved_pose: " << solved_pose.m_coords[0] << ",";                      // solved_pose_x
    file << solved_pose.m_coords[1] << ",";                                         // solved_pose_y
    file << solved_pose.m_coords[2] << ",";                                         // solved_pose_z
    file << solved_pose.m_quat[0] << ",";                                           // solved_pose_qw
    file << solved_pose.m_quat[1] << ",";                                           // solved_pose_qx
    file << solved_pose.m_quat[2] << ",";                                           // solved_pose_qy
    file << solved_pose.m_quat[3] << ", ";                                          // solved_pose_qz
    file << "err_dist: " << results->get_distance_error() << ", ";                  // err_dist
    file << "err_rot_angle: " << results->get_angular_error() << ", ";              // err_rot_angle
    file << "num_iterations: " << results->get_num_iterations() << ", ";            // num_iterations
    file << "total_time: " << results->get_total_time() << ", ";                    // total_time
    file << "convergence: " << results->has_converged()<< ", ";                     // convergence
    file << "image_scale: " << pyramid_height;										// image_scale
    file << endl;
    file << "=============================================================" << endl;

    double solved_pose_x = solved_pose.m_coords[0];
    double solved_pose_y = solved_pose.m_coords[1];
    double solved_pose_z = solved_pose.m_coords[2];
    double solved_pose_qw = solved_pose.m_quat[0];
    double solved_pose_qx = solved_pose.m_quat[1];
    double solved_pose_qy = solved_pose.m_quat[2];
    double solved_pose_qz = solved_pose.m_quat[3];

    for (int i = pyramid_height - 1; i > 0; i--) {

	    params.exact_initial_pose = true;
	    params.solved_pose_lower_scale.m_coords[0] = solved_pose_x;
	    params.solved_pose_lower_scale.m_coords[1] = solved_pose_y;
	    params.solved_pose_lower_scale.m_coords[2] = solved_pose_z;
	    params.solved_pose_lower_scale.m_quat[0] = solved_pose_qw;
	    params.solved_pose_lower_scale.m_quat[1] = solved_pose_qx;
	    params.solved_pose_lower_scale.m_quat[2] = solved_pose_qy;
	    params.solved_pose_lower_scale.m_quat[3] = solved_pose_qz;

	    // start the solving process
	    const posest::ExecutionResults *results = params.posest_start(dataset, i);

	    // get error after convergence with respect to ground truth
	    const CArrayDouble<3> &err_pos = results->get_position_error();
	    const CQuaternionDouble &err_rot = results->get_rotation_error();

	    // get solved pose
	    const CPose3DQuat &solved_pose = results->get_solved_pose();

	    file << "Layer: " << i << endl;
	   	file << "initial_pose: " << params.solved_pose_lower_scale.m_coords[0] << ",";	// initial_pose_x
	   	file << params.solved_pose_lower_scale.m_coords[1] << ",";						// initial_pose_y
	   	file << params.solved_pose_lower_scale.m_coords[2] << ",";						// initial_pose_z
	   	file << params.solved_pose_lower_scale.m_quat[0] << ",";						// initial_pose_qw
	   	file << params.solved_pose_lower_scale.m_quat[1] << ",";						// initial_pose_qx
	   	file << params.solved_pose_lower_scale.m_quat[2] << ",";						// initial_pose_qy
	   	file << params.solved_pose_lower_scale.m_quat[3] << ", ";						// initial_pose_qz
    	file << "err: " << err_pos[0] << ",";                                           // err_x
   	 	file << err_pos[1] << ",";                                                      // err_y
    	file << err_pos[2] << ",";                                                      // err_z
    	file << err_rot[0] << ",";                                                      // err_qw
   		file << err_rot[1] << ",";                                                      // err_qx
    	file << err_rot[2] << ",";                                                      // err_qy
    	file << err_rot[3] << ", ";                                                     // err_qz
    	file << "solved_pose: " << solved_pose.m_coords[0] << ",";                      // solved_pose_x
   		file << solved_pose.m_coords[1] << ",";                                         // solved_pose_y
    	file << solved_pose.m_coords[2] << ",";                                         // solved_pose_z
   		file << solved_pose.m_quat[0] << ",";                                           // solved_pose_qw
    	file << solved_pose.m_quat[1] << ",";                                           // solved_pose_qx
    	file << solved_pose.m_quat[2] << ",";                                           // solved_pose_qy
    	file << solved_pose.m_quat[3] << ", ";                                          // solved_pose_qz
        file << "err_dist: " << results->get_distance_error() << ", ";                  // err_dist
    	file << "err_rot_angle: " << results->get_angular_error() << ", ";              // err_rot_angle
    	file << "num_iterations: " << results->get_num_iterations() << ", ";            // num_iterations
    	file << "total_time: " << results->get_total_time() << ", ";                    // total_time
    	file << "convergence: " << results->has_converged()<< ", ";                     // convergence
    	file << "image_scale: " << i;										// image_scale
    	file << endl;
    	file << "=============================================================" << endl;

    	solved_pose_x = solved_pose.m_coords[0];
    	solved_pose_y = solved_pose.m_coords[1];
    	solved_pose_z = solved_pose.m_coords[2];
    	solved_pose_qw = solved_pose.m_quat[0];
    	solved_pose_qx = solved_pose.m_quat[1];
    	solved_pose_qy = solved_pose.m_quat[2];
    	solved_pose_qz = solved_pose.m_quat[3];
    }

    // print summary
    cout << "-------------------------------------------------------------" << endl;
    cout << " finished solving for ref_img: "
         << params.ref_img_index
         << ", blurred_img: " << params.blurred_img_index << endl;
    cout << "=============================================================" << endl;
}