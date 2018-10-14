/**
 * Warp a sharp reference image together with its depth map into a camera at a different pose.
 * The other pose is taken from ground truth of the target image.
 */
#include <posest/posest.h>
#include <posest/dataset.h>
#include <posest/reprojector.h>
#include <opencv2/imgproc.hpp>
#include <string>

int main(int argc, char *args[]) {
    // minimal args check
    if (argc != 7) {
        std::cerr << "reproject warps a sharp reference image together with its perturbed depth map into " << std::endl;
        std::cerr << "a camera at a different pose. The amount of perturbation can be controlled with " << std::endl;
        std::cerr << "sigma." << std::endl;
        std::cerr << std::endl;
        std::cerr << "Usage: " << args[0] << " [dataset] [cam_index] [ref_img_index] [target_img_index] "
                                             "[sigma] [output_img]" << std::endl;
        return 1;
    }

    // read command libe arguments and images
    posest::Dataset dataset(args[1]);
    dataset.read();
    const int cam_index = atoi(args[2]);
    const int ref_img_index = atoi(args[3]);
    const int reproj_img_index = atoi(args[4]);
    const double sigma = atof(args[5]);
    const std::string reproj_img_path(args[6]);
    //const cv::Mat_<uchar> &ref_img = dataset.readSharpImage(ref_img_index, cam_index);
    const cv::Mat_<uchar> &ref_img = dataset.readSharpScaledImage(img_index, cam_index, 2); 

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
            dataset.getPose(ref_img_index, cam_index),
            2);

    // generate reprojected image
    cv::Mat_<uchar> reproj_img(ref_img.size());
    cv::Mat_<bool> reproj_mask(ref_img.size());
    reprojector.reproject(dataset.getPose(reproj_img_index, cam_index), reproj_img, reproj_mask);

    // colorize empty pixels before writing file to disk
    cv::Mat output;
    cv::cvtColor(reproj_img, output, CV_GRAY2RGB);
    for (auto &element : reproj_mask) {
        element = !element;
    }
    output.setTo(cv::Scalar(160, 250, 255), reproj_mask);  // some yellowish color
    cv::imwrite(reproj_img_path, output);

    return 0;
}
