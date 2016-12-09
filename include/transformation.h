#ifndef _DGFACESDK_TRANSFORMATION_H_
#define _DGFACESDK_TRANSFORMATION_H_

#include "common.h"
#include "face_inf.h"
namespace DGFace{
enum class transform_method : unsigned char{
	CDNN,
	CDNN_CAFFE,
};

class Transformation{
	public:
		Transformation();
		virtual ~Transformation(void);
		void transform(const cv::Mat& img, const AlignResult& src_align, 
						cv::Mat& transformed_img, AlignResult& transformed_align);
	protected:
		virtual void transform_impl(const cv::Mat& img, const AlignResult& src_align, 
									cv::Mat& transformed_img, AlignResult& transformed_align) = 0;
		virtual void transform_impl(const cv::Mat& img, const LandMarkInfo& src_landmark_info,
									cv::Mat& transformed_img, LandMarkInfo& transformed_landmark_info) = 0;
		virtual void transform_impl(const cv::Mat& img, const std::vector<double>& src_landmarks,
									cv::Mat& transformed_img, std::vector<double>& transformed_landmarks) = 0;
};

//Transformation *create_transformation(const std::string &prefix = std::string());
Transformation *create_transformation(const transform_method& method, const std::string& model_dir);

}
#endif
