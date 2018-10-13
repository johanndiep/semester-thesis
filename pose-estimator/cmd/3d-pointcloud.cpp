/**
 * 3d-pointcloud takes a sharp reference image together with its depth map and writes a
 * 3D point cloud file where the image is unprojected back to 3D space.
 * The 3D point coordinates are given in the world coordinate frame.
 *
 * The output file can easily be inspected in meshlab.
 */
#include <posest/dataset.h>
#include <posest/reprojector.h>

int main(int argc, char *args[]) {
    // minimal args check
    if (argc != 6) {
        std::cerr << "3d-pointcloud takes a sharp reference image together with its depth map and writes" << std::endl;
        std::cerr << "a 3D point cloud file where the image is unprojected back to 3D space." << std::endl;
        std::cerr << "Use sigma to perturb the depth map." << std::endl;
        std::cerr << std::endl;
        std::cerr << "Usage: " << args[0] << " [dataset] [cam_index] [img_index] [sigma] [output]" << std::endl;
        return 1;
    }

    // read command line arguments and images
    posest::Dataset dataset(args[1]);
    dataset.read();
    const int cam_index = atoi(args[2]);
    const int img_index = atoi(args[3]);
    const double sigma = atof(args[4]);
    const cv::Mat_<uchar> &ref_img = dataset.readSharpImage(img_index, cam_index);

    cv::Mat_<double> depth_map;
    if (sigma == 0) {
        depth_map = dataset.readDepthImage(img_index, cam_index);
    } else {
        depth_map = dataset.readDepthImage(img_index, cam_index, sigma);
    }

    // setup reprojector
    posest::ReprojectorImpl reprojector(
            dataset.getInternalCalibration(cam_index),
            ref_img,
            depth_map,
            dataset.getPose(img_index, cam_index));

    // get the 3D point cloud
    auto points3D = reprojector.getPoints3D();

    // write the 3D point cloud to a file which can be read in meshlab
    std::ofstream points3D_file;
    points3D_file.open(args[5]);
    auto it = ref_img.begin();
    for (auto p : points3D) {
        int grey = *(it++);
        points3D_file << p.x() << ";" << p.y() << ";" << p.z() << ";"
        << grey << ";" << grey << ";" << grey << std::endl;
    }
    points3D_file.close();

    return 0;
}
