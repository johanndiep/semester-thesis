/**
 * Use blur to output an artificially blurred image taken at a pose of an other image.
 */
#include <posest/dataset.h>
#include <posest/reprojector.h>
#include <opencv2/imgproc.hpp>
#include <posest/blurrer.h>
#include <string>

int main(int argc, char *args[]) {
    // minimal args check
    if (argc != 8) {
        std::cerr << "blur creates an artificially blurred image taken at a pose" << std::endl;
        std::cerr << "of an other image in the dataset. Use sigma to perturb the depth map." << std::endl;
        std::cerr << "" << std::endl;
        std::cerr << "Usage: " << args[0]
                  << " [dataset] [cam_index] [ref_img_index] [target_img_index] [n_images] [sigma] [output_img]"
                  << std::endl;
        return 1;
    }

    // read command line arguments and images
    posest::Dataset dataset(args[1]);
    dataset.read();
    const int cam_index = atoi(args[2]);
    const int ref_img_index = atoi(args[3]);
    const int blur_img_index = atoi(args[4]);
    const int n_images = atoi(args[5]);
    const double sigma = atof(args[6]);
    const std::string blur_img_path(args[7]);
    //const cv::Mat_<uchar> &ref_img = dataset.readSharpImage(ref_img_index, cam_index);
    const cv::Mat_<uchar> &ref_img = dataset.readSharpScaledImage(ref_img_index, cam_index, 2); 
    const mrpt::poses::CPose3DQuat &ref_pose = dataset.getPose(ref_img_index, cam_index);

    cv::Mat_<double> depth_map;
    if (sigma == 0) {
        depth_map = dataset.readDepthImage(ref_img_index, cam_index);
        depth_map = dataset.readScaledDepthImage(depth_map, 2);
    } else {
        depth_map = dataset.readDepthImage(ref_img_index, cam_index, sigma);
        depth_map = dataset.readScaledDepthImage(depth_map, 2);
    }

    // setup reprojector
    posest::ReprojectorImpl reprojector(
            dataset.getInternalCalibration(cam_index),
            ref_img,
            depth_map,
            ref_pose, 2);

    // setup blurrer
    posest::BlurrerImpl blurrer(ref_pose,
                                dataset.getTimestamp(ref_img_index, cam_index),
                                dataset.getTimestamp(blur_img_index, cam_index),
                                dataset.getExposureTime(), n_images, ref_img, reprojector, 2);

    // generate the blurred image
    cv::Mat_<uchar> blur_img(ref_img.size());
    cv::Mat_<bool> blur_mask(ref_img.size());
    blurrer.blur(dataset.getPose(blur_img_index, cam_index), blur_img, blur_mask);

    // colorize empty pixels before writing file to disk
    cv::Mat output;
    cv::cvtColor(blur_img, output, CV_GRAY2RGB);
    for (auto &element : blur_mask) {
        element = !element;
    }
    output.setTo(cv::Scalar(160, 250, 255), blur_mask);  // some yellowish color
    cv::imwrite(blur_img_path, output);

    return 0;
}
