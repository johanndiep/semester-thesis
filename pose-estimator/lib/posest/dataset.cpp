#include <utility>
#include "posest/dataset.h"
#include <cmath>
#include <random>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

using std::map;
using std::string;
using std::ifstream;
using std::vector;
using mrpt::poses::CPose3DQuat;

/**
 * Returns true if the string str starts with needle
 * @param str
 * @param needle
 * @return
 */
inline bool starts_with(string &str, const char *needle) {
    // std::string::find returns 0 if needle is found at starting
    return str.find(needle) == 0;
}

posest::Dataset::Dataset(const string &base_path) : base_path(base_path) {}

void posest::Dataset::readCameraCalibration(map<string, CameraCalibration> &cams) {
    const string extrinsics_filename = "/extrinsics.txt";
    const string intrinsics_filename = "/intrinsics.txt";
    ifstream intrinsics_file(base_path + intrinsics_filename);
    ifstream extrinsics_file(base_path + extrinsics_filename);

    // test if files can be read
    if (intrinsics_file.fail() || extrinsics_file.fail()) {
        throw std::runtime_error("can't read camera calibration files");
    }

    // read and parse intrinsics
    string current_cam;
    string line;
    while (getline(intrinsics_file, line)) {
        std::istringstream line_reader(line);

        if (starts_with(line, "devName ")) {
            string label;
            line_reader >> label >> current_cam;
            cams.insert(make_pair(current_cam, CameraCalibration()));
        }

        if (starts_with(line, "K ")) {
            InternalCalibration &calibration = cams[current_cam].internal;
            string label;
            line_reader >> label >> calibration.fx >> calibration.fy >> calibration.cx >> calibration.cy;
        }

        if (starts_with(line, "resolution ")) {
            cv::Size &resolution = cams[current_cam].resolution;
            string label;
            line_reader >> label >> resolution.width >> resolution.height;
        }
    }

    current_cam = "";

    // read and parse extrinsics
    while (getline(extrinsics_file, line)) {
        std::istringstream line_reader(line);

        if (starts_with(line, "devName ")) {
            string label;
            line_reader >> label >> current_cam;
            if (cams.find(current_cam) == cams.end()) {
                throw std::runtime_error("no extrinsic configuration found for camera");
            }
        }

        if (starts_with(line, "T ")) {
            mrpt::poses::CPose3DQuat &external = cams[current_cam].external;
            double qw, qx, qy, qz, x, y, z;
            double qw_new, qx_new, qy_new, qz_new;

            string label;
            line_reader >> label >> qw >> qx >> qy >> qz >> x >> y >> z;
            // make sure quaternion is normalized
            qw_new = qw / sqrt(qw * qw + qx * qx + qy * qy + qz * qz);
            qx_new = qx / sqrt(qw * qw + qx * qx + qy * qy + qz * qz);
            qy_new = qy / sqrt(qw * qw + qx * qx + qy * qy + qz * qz);
            qz_new = qz / sqrt(qw * qw + qx * qx + qy * qy + qz * qz);
            qw = qw_new;
            qx = qx_new;
            qy = qy_new;
            qz = qz_new;
            external.m_quat.r(qw);
            external.m_quat.x(qx);
            external.m_quat.y(qy);
            external.m_quat.z(qz);
            external.m_coords[0] = x;
            external.m_coords[1] = y;
            external.m_coords[2] = z;
        }

        if (starts_with(line, "scalingFactor ")) {
            string label;
            line_reader >> label >> cams[current_cam].scaling_factor;
        }
    }

    // apply scaling factor, everything in meter
    for (auto &cam : cams) {
        double s = cam.second.scaling_factor;
        cam.second.external.m_coords[0] *= s;
        cam.second.external.m_coords[1] *= s;
        cam.second.external.m_coords[2] *= s;
    }
}

void posest::Dataset::readCameraTrajectory(const CPose3DQuat &camera_offset,
                                           Trajectory &trajectory) {
    const string trajectory_filename = "/groundTruthPoseVel_imu.txt";
    ifstream trajectory_file(base_path + trajectory_filename);

    if (trajectory_file.fail()) {
        throw std::runtime_error("can't read ground truth camera trajectory file");
    }

    string line;
    while (getline(trajectory_file, line)) {
        if (starts_with(line, "#")) {
            continue;
        }
        std::istringstream line_reader(line);
        Waypoint wpt;
        double qw, qx, qy, qz, x, y, z;
        line_reader >> wpt.time >> qw >> qx >> qy >> qz >> x >> y >> z;
        wpt.pose.m_quat.r(qw);
        wpt.pose.m_quat.x(qx);
        wpt.pose.m_quat.y(qy);
        wpt.pose.m_quat.z(qz);
        wpt.pose.m_coords[0] = x;
        wpt.pose.m_coords[1] = y;
        wpt.pose.m_coords[2] = z;
        wpt.pose += camera_offset;
        trajectory.push_back(wpt);
    }
}

void posest::Dataset::readImageLogs() {
    const string img_logs_filename = "/imageLogs.txt";
    ifstream trajectory_file(base_path + img_logs_filename);

    if (trajectory_file.fail()) {
        throw std::runtime_error("can't read ground image logs file");
    }

    string line;
    while (getline(trajectory_file, line)) {
        if (starts_with(line, "#")) {
            continue;
        }

        std::istringstream line_reader(line);
        std::string img_type_str;
        ImageLogEntry log_entry;
        line_reader >> log_entry.timestamp >> log_entry.seq_num >> img_type_str >> log_entry.cam_dev_name
                    >> log_entry.exposure_time >> log_entry.path_to_image;
        if (img_type_str == "blur") {
            log_entry.img_type = blur;
        } else if (img_type_str == "normal") {
            log_entry.img_type = normal;
        } else if (img_type_str == "depth") {
            log_entry.img_type = depth;
        } else if (img_type_str == "rgb") {
            log_entry.img_type = rgb;
        } else {
            throw std::runtime_error("unknown image type read in imageLogs.txt file");
        }
        this->image_log.push_back(log_entry);
    }
}

void posest::Dataset::read() {
    map<string, CameraCalibration> cams;
    readCameraCalibration(cams);
    readImageLogs();

    // for each camera read its trajectory and fill cameras list
    for (auto &cam : cams) {
        CameraCalibration &cc = cam.second;
        Trajectory t;
        readCameraTrajectory(cc.external, t);
        cameras.push_back(cam.first);
        calibrations.push_back(cc);
        trajectories.push_back(t);
    }
}

const std::vector<std::string> &posest::Dataset::getCameras() const {
    return this->cameras;
}

cv::Mat_<double> posest::Dataset::readDepthImage(const int index, const int cam_index) const {
    string cam_name = this->cameras[cam_index];
    string path = this->base_path + "/depth/" + cam_name + "/" + std::to_string(index) + ".exr";

    ifstream depth_file(path);
    if (depth_file.fail()) {
        throw std::runtime_error("can't read depth image: " + path);
    }

    const cv::Size &resolution = this->calibrations[cam_index].resolution;
    cv::Mat_<double> depth_map(resolution);
    int row = 0;
    string line;

    while (getline(depth_file, line)) {
        std::istringstream line_reader(line);
        for (int col = 0; col < resolution.width; ++col) {
            line_reader >> depth_map.at<double>(row, col);
            // value 65504 means infinitely far away, therefore we just set the value to 0
            if (depth_map.at<double>(row, col) >= 65504) {
                depth_map.at<double>(row, col) = 0;
            }
        }
        row++;
    }
    return depth_map;
}

cv::Mat_<double> posest::Dataset::readDepthImage(int img_index, int cam_index, double sigma_0) const {
    // Read sharp ref image
    cv::Mat_<uchar> ref_img = readSharpImage(img_index, cam_index);
    // Read depth map
    cv::Mat_<double> depth_map = readDepthImage(img_index, cam_index);
    // Perturb depth map depending on gradient of ref image and sigma_0
    // Compute dx and dy derivatives
    cv::Mat dx, dy;
    cv::Sobel(ref_img, dx, CV_32F, 1, 0);
    cv::Sobel(ref_img, dy, CV_32F, 0, 1);

    // Compute gradient magnitude
    cv::Mat grad_magn_matrix;
    magnitude(dx, dy, grad_magn_matrix);

    std::default_random_engine rng(static_cast<unsigned int>(time(nullptr)));

    double sigma;
    double noise;
    double grad_magn;
    double eps = 0.15;
    // Build output depth map for each pixel
    for (int row = 0; row < ref_img.rows; row++) {
        for (int col = 0; col < ref_img.cols; col++) {
            // Extract gradient magnitude at that pixel
            grad_magn = grad_magn_matrix.at<double>(row, col);

            sigma = sigma_0 * exp(-grad_magn / 1e10);

            // Compute additive Gaussian noise
            std::normal_distribution<double> distribution(0.0, sigma);
            noise = distribution(rng);
            // Assign perturbed value to output depth map
            depth_map.at<double>(row, col) *= (1 + noise);
        }
    }
    return depth_map;
}

cv::Mat_<uchar> posest::Dataset::readBlurredImage(const int index, const int cam_index) const {
    string cam_name = this->cameras[cam_index];
    string path = base_path + "/blurred/" + cam_name + "/" + std::to_string(index) + ".png";
    return cv::imread(path, cv::IMREAD_GRAYSCALE);
}

cv::Mat_<uchar> posest::Dataset::readSharpImage(const int index, const int cam_index) const {
    string cam_name = this->cameras[cam_index];
    string path = base_path + "/rgb/" + cam_name + "/" + std::to_string(index) + ".png";
    return cv::imread(path, cv::IMREAD_GRAYSCALE);
}

double posest::Dataset::getTimestamp(int img_index, int cam_index) const {
    double timestamp;
    bool found = false;
    for (auto img_log : this->image_log) {
        if (img_log.cam_dev_name == this->cameras[cam_index] &&
            img_log.seq_num == img_index) {
            timestamp = img_log.timestamp;
            found = true;
            break;
        }
    }
    if (!found) {
        throw std::runtime_error("could not find timestamp for given index");
    }
    return timestamp;
}

const CPose3DQuat &posest::Dataset::getPose(const int index, const int cam_index) const {
    double timestamp = getTimestamp(index, cam_index);
    return getWaypointForTime(cam_index, timestamp).pose;
}

const posest::InternalCalibration &posest::Dataset::getInternalCalibration(int cam_index) const {
    return this->calibrations[cam_index].internal;
}

const posest::Dataset::Waypoint &posest::Dataset::getWaypointForTime(int cam_index, double time) const {
    for (auto &w : trajectories[cam_index]) {
        if (w.time >= time - 1e-4) {
            return w;
        }
    }
    throw std::runtime_error("no waypoint found for time: " + std::to_string(time));
}

double posest::Dataset::getExposureTime() const {
    // (!) this assumes that exposure time for all blurred images is equal
    // and just returns the exposure time of the first blurred image
    for (auto img_log : this->image_log) {
        if (img_log.img_type == blur) {
            return img_log.exposure_time;
        }
    }
    return 0;
}

const mrpt::poses::CPose3DQuat &posest::Dataset::getPoseAtTime(double time, int cam_index) const {
    return getWaypointForTime(cam_index, time).pose;
}
