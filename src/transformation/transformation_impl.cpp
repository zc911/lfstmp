#include <transformation/trans_cdnn.h>
#include <transformation/trans_cdnn_caffe.h>
#include <config.h>
#include <stdexcept>
#include "face_para.h"
#include "dgface_utils.h"


using namespace std;
using namespace cv;
namespace DGFace {
//----------------Cdnn---------------//
CdnnTransformation::CdnnTransformation() {
}

CdnnTransformation::~CdnnTransformation() {
}

void CdnnTransformation::transform_impl(const Mat& img, const AlignResult& src_align,
										Mat& transformed_img, AlignResult& transformed_align) {
	transformed_align = src_align;
	vector<double> src_landmarks_vec, transformed_landmarks_vec;
	cvtLandmarks(src_align.landmarks, src_landmarks_vec);

	transform_impl(img, src_landmarks_vec, transformed_img, transformed_landmarks_vec);

	if(transformed_landmarks_vec.size() != src_landmarks_vec.size()) {
		transformed_align.landmarks.clear();
	}
	cvtLandmarks(transformed_landmarks_vec, transformed_align.landmarks);
}

void CdnnTransformation::transform_impl(const Mat& img, const LandMarkInfo& src_landmark_info,
										Mat& transformed_img, LandMarkInfo& transformed_landmark_info) {
	transformed_landmark_info = src_landmark_info;
	vector<double> src_landmarks_vec, transformed_landmarks_vec;
	cvtLandmarks(src_landmark_info.landmarks, src_landmarks_vec);

	transform_impl(img, src_landmarks_vec, transformed_img, transformed_landmarks_vec);

	if(transformed_landmarks_vec.size() != src_landmarks_vec.size()) {
		transformed_landmark_info.landmarks.clear();
	}
	cvtLandmarks(transformed_landmarks_vec, transformed_landmark_info.landmarks);
}

void CdnnTransformation::transform_impl(const Mat& img, const vector<double>& src_landmarks,
										Mat& transformed_img, vector<double>& transformed_landmarks) {
	transformed_landmarks.clear();

	IplImage ipl_image = img;
	IplImage* srcImage = &ipl_image;
	IplImage* dstImage = NULL;

	bool ret = _transformer.AffineTransform(srcImage, dstImage, src_landmarks, transformed_landmarks);
	if(!ret) {
		transformed_landmarks.clear();
	}
	
	transformed_img = cvarrToMat(dstImage, true);
	cvReleaseImage(&dstImage);
}

//--------------------------cdnn_caffe--------------------//
CdnnCaffeTransformation::CdnnCaffeTransformation() {
}

CdnnCaffeTransformation::~CdnnCaffeTransformation() {
}

void CdnnCaffeTransformation::transform_impl(const Mat& img, const AlignResult& src_align,
											Mat& transformed_img, AlignResult& transformed_align) {
	transformed_align = src_align;
	LandMarkInfo src_landmarks_info, transformed_landmark_info;
	cvtLandmarks(src_align.landmarks, src_landmarks_info.landmarks);

	transform_impl(img, src_landmarks_info,transformed_img, transformed_landmark_info);

	cvtLandmarks(transformed_landmark_info.landmarks, transformed_align.landmarks);
}

void CdnnCaffeTransformation::transform_impl(const Mat& img, const vector<double>& src_landmarks,
											Mat& transformed_img, vector<double>& transformed_landmarks) {
	LandMarkInfo src_landmarks_info, transformed_landmark_info;
	cvtLandmarks(src_landmarks, src_landmarks_info.landmarks);

	transform_impl(img, src_landmarks_info,transformed_img, transformed_landmark_info);

	cvtLandmarks(transformed_landmark_info.landmarks, transformed_landmarks);
}

void CdnnCaffeTransformation::transform_impl(const Mat& img, const LandMarkInfo& src_landmarks,
											Mat& transformed_img, LandMarkInfo& transformed_landmarks) {

	IplImage ipl_image = img;
	IplImage* srcImage = &ipl_image;
	IplImage* dstImage = NULL;
	FacePara param;

	_transformer.GetWarpImage(srcImage, param, src_landmarks, &dstImage, transformed_landmarks);

	transformed_img = cvarrToMat(dstImage, true);
	cvReleaseImage(&dstImage);
}

////////////////////////////////////////////////////////////////////
Transformation *create_transformation(const string &prefix) {
	Config *config = Config::instance();
	string type = config->GetConfig<string>(prefix + "transformation", "cdnn");
	if(type == "cdnn") {
		return new CdnnTransformation();
	} else if(type == "cdnn_caffe") {
		return new CdnnCaffeTransformation();
	}
	throw new runtime_error("unknown transformation");
}

}
