#ifndef _dgfaceskd_trans_cdnn_h_
#define _dgfaceskd_trans_cdnn_h_
#include <transformation.h>
#include "face_inf.h"
#include "ExtractMPMetricFeature.h"
namespace DGFace {

class CdnnTransformation : public Transformation{
	public:
		CdnnTransformation();
		virtual ~CdnnTransformation(void);

		void transform_impl(const cv::Mat& img, const AlignResult& src_align,
							cv::Mat& transformed_img, AlignResult& transformed_align);
		void transform_impl(const cv::Mat& img, const LandMarkInfo& src_landmark_info,
							cv::Mat& transformed_img, LandMarkInfo& transformed_landmark_info);
		void transform_impl(const cv::Mat& img, const std::vector<double>& src_landmarks,
							cv::Mat& transformed_img, std::vector<double>& transformed_landmarks);
	private:
		Cdnn::MPMetricFeature::ExtractMPMetricFeature _transformer;
		
};

}
#endif
