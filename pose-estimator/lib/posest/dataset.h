#ifndef LIB_POSEST_DATASET_H_
#define LIB_POSEST_DATASET_H_

#include <string>
#include <map>
#include <ostream>
#include <vector>
#include "mrpt/poses.h"
#include "posest/posest.h"

namespace posest {

class Dataset {
    /**
     * A point together with a timestamp of a camera trajectory
     */
    struct Waypoint {
        double time;
        mrpt::poses::CPose3DQuat pose;
    };
    typedef std::vector<Waypoint> Trajectory;

    /**
     * CameraCalibration contains the internal calibration (basically the K matrix) and
     * the external pose which is the offset between the vehicle body frame and the camera local frame
     */
    struct CameraCalibration {
        mrpt::poses::CPose3DQuat external;
        InternalCalibration internal;

        /**
         * Image resolution
         */
        cv::Size resolution;

        /**
         * Uhe external position vector needs to be multiplied by this number to receive meters.
         * Usually this is set to 0.01 which means that external coordinates are given in cm.
         */
        double scaling_factor;
    };

    /**
     * Image types as defined in imageLogs.txt
     */
    enum ImageType {
        blur, rgb, depth, normal
    };

    /**
     * A row of the imageLogs.txt file
     */
    struct ImageLogEntry {
        // timestamp of closing shutter
        double timestamp;
        // exposure time given in seconds
        double exposure_time;
        // n-th image in the sequence
        int seq_num;
        // img_type
        ImageType img_type;
        std::string cam_dev_name;
        std::string path_to_image;
    };

    std::vector<CameraCalibration> calibrations;
    std::vector<Trajectory> trajectories;
    std::vector<std::string> cameras;
    std::vector<ImageLogEntry> image_log;
    const std::string base_path;
    
    void readCameraCalibration(std::map<std::string, CameraCalibration> &cams);

    void readCameraTrajectory(const mrpt::poses::CPose3DQuat &camera_offset, Trajectory &trajectory);

    void readImageLogs();

    const Waypoint &getWaypointForTime(int cam_index, double time) const;


 public:
    /**
     * Create a dataset wich is located at base_path.
     * Call read to actually read the files.
     * @param base_path
     */
    explicit Dataset(const std::string &base_path);

    /**
     * Read dataset files with camera trajectory and calibrations
     */
    void read();

    /**
     * getCameras returns the camera names as vector
     * usually dev0 and dev1
     * @return
     */
    const std::vector<std::string> &getCameras() const;

    /**
     * Return camera calibration for camera cam_index.
     * The calibration is returned for camera getCameras()[cam_index]
     * @param cam_index
     * @return
     */
    const InternalCalibration &getInternalCalibration(int cam_index) const;

    /**
     * Read a depth map for image img_index taken by camera cam_index
     * @param img_index usually it starts from 1 (and not 0)
     * @param cam_index use camera getCameras()[cam_index]
     */
    cv::Mat_<double> readDepthImage(int img_index, int cam_index) const;

    /**
     * Read depth map and perturbe values with an additive Gaussian noise with standard deviation sigma and zero mean.
     *
     * sigma is computed as follows:
     *
     *      sigma = sigma_0 * exp(-gradient_magnitude(x,y)) ,
     *      
     * @param img_index index of the input single-channel image in the dataset;
     * @param cam_index cam_index: index of the camera in the dataset;
     * @param sigma_0: sigma_0 in the formula above;
     * @return depth_map output perturbed depth map.
     */
    cv::Mat_<double> readDepthImage(int img_index, int cam_index, double sigma_0) const;

    /**
     * Read a blurred image from the dataset
     * @param img_index usually it starts from 1 (and not 0)
     * @param cam_index use camera getCameras()[cam_index]
     * @return
     */
    cv::Mat_<uchar> readBlurredImage(int img_index, int cam_index) const;

    /**
     * Read a blurred image from the dataset and downscale it by a factor
     * @param img_index usually it starts from 1 (and not 0)
     * @param cam_index use camera getCameras()[cam_index]
     * @param image_scale defines the downscale (i.e. 2 means divide resolution by 2)
     * @return
     */
    cv::Mat_<uchar> readBlurredScaledImage(int img_index, int cam_index) const;

    /**
     * Read a sharp image from the dataset (converted to grayscale)
     * @param img_index usually it starts from 1 (and not 0)
     * @param cam_index use camera getCameras()[cam_index]
     * @return
     */
    cv::Mat_<uchar> readSharpImage(int img_index, int cam_index) const;

    /**
    * Read a sharp image from the dataset and downscale it by a factor (converted to grayscale)
    * @param img_index usually it starts from 1 (and not 0)
    * @param cam_index use camera getCameras()[can_index]
    * @param image_scale defines the downscale (i.e. 2 means divide resolution by 2)
    * @return
    */
    cv::Mat_<uchar> readSharpScaledImage(int img_index, int cam_index, double image_scale) const;

    /**
     * Get the pose of a camera when a certain image was taken.
     * This is always the pose of the closing shutter.
     * @param img_index usually it starts from 1 (and not 0)
     * @param cam_index use camera getCameras()[cam_index]
     * @return camera pose given in world coordinates
     */
    const mrpt::poses::CPose3DQuat &getPose(int img_index, int cam_index) const;

    /**
     * Get the pose of a camera at a certain timestamp.
     * @param time usually it starts from 1 (and not 0)
     * @param cam_index use camera getCameras()[cam_index]
     * @return camera pose given in world coordinates
     */
    const mrpt::poses::CPose3DQuat &getPoseAtTime(double time, int cam_index) const;

    /**
     * Returns the exposure time of blurred images.
     * This assumes that exposure times for all blurred images are equal.
     * @return exposure time in seconds
     */
    double getExposureTime() const;

    /**
     * Get timestamp of an image.
     * This always returns the timestamp of the closing shutter.
     * @param img_index usually it starts from 1 (and not 0)
     * @param cam_index use camera getCameras()[cam_index]
     * @return timestamp in seconds
     */
    double getTimestamp(int img_index, int cam_index) const;
};

}  // namespace posest

#endif  // LIB_POSEST_DATASET_H_
