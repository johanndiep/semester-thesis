// Example program
#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>

namespace posest {

class Dataset {

    const std::string base_path;
	double image_scale;

public:

	Dataset(const std::string &base_path, const double image_scale) : base_path(base_path), image_scale(image_scale) {}

	cv::Mat_<uchar> readSharpScaledImage() const {
		std::string path = base_path + "/rgb/" + "cam0" + "/" + "1.png";
		cv::Mat_<uchar> LoadedImage = cv::imread(path, cv::IMREAD_GRAYSCALE);
		cv::resize(LoadedImage, LoadedImage, cv::Size(LoadedImage.cols/image_scale, LoadedImage.rows/image_scale));
		return LoadedImage;
	}

};

}

int main(int argc, const char** argv)
{

	posest::Dataset dataset("/home/johann/motion-blur-cam-pose-tracker/semester-thesis/RelisticRendering-dataset", 4);

	cv::Mat_<uchar> ScaledImage = dataset.readSharpScaledImage();
 	cv::imwrite("Step1.JPG", ScaledImage);


 	return 0;
}
