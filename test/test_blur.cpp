// file: test_blur.cpp
// date: 20170207
// email: xinzhao@deepglint.com

#include <iostream>
#include <opencv2/opencv.hpp>
#include <detector.h>
#include <quality.h>

using namespace cv;
using namespace std;
using namespace DGFace;

int main( int argc, char* argv[] ) {
	cout<<"hello world"<<endl;
	string img_loc = "test/images/lena.jpg";
	const float ground_truth = -0.230997f;
	cv::Mat m = cv::imread(img_loc);
	imshow("m", m);
	waitKey(0);

	Detector  *detector = create_detector(det_method::SSD, "data/model/detector/ssd/0.0.4", 0);
	vector<DetectResult> detect_results;

	std::vector<cv::Mat> ms = {m};

	detector->detect(ms, detect_results);
	cout << "detect " << detect_results[0].boundingBox.size() << "faces from: " << img_loc << endl;
	Rect r;
	Mat face_img = m(detect_results[0].boundingBox[0].second.boundingRect());
	imshow("face_img", face_img);
	waitKey(0);

	Quality *quality = create_quality(quality_method::LENET_BLUR, "data/model/quality/blur/0.1.0", 0, false, 4);
//	Quality *quality = create_quality_with_global_dir(quality_method::LENET_BLUR, "data", 0, false);
	float blur = quality->quality(face_img);
	printf("predict blur = %f\n", blur);
	if (fabs(blur-ground_truth)>1e-6) {
		printf("ERROR: ground_truth = %f, predict blur = %f\n", ground_truth, blur);
	}

	vector<Mat> imgs;
	imgs.push_back(face_img);
	imgs.push_back(face_img);
	imgs.push_back(face_img);
	imgs.push_back(face_img);
	vector<float> blurs;
	quality->quality(imgs, blurs);
	for (size_t i=0; i<blurs.size(); i++ ) {
		printf("blurs[%d] = %f\n", i, blurs[i]);
	}
	return 0;
}
