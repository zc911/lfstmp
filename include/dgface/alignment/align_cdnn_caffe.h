#ifndef _dgfacesdk_alignment_cdnn_caffe_h_
#define _dgfacesdk_alignment_cdnn_caffe_h_
#include <string>

#include <alignment.h>
#include "FaceAlign.h"
#include "FaceWarp.h"
#include "ExtractMPMetricFeature.h"
namespace DGFace{

class CdnnCaffeAlignment : public Alignment{
	public:
		CdnnCaffeAlignment(std::vector<int> face_size, std::string config_file, std::string warp_config_file, int gpu_id);
		virtual ~CdnnCaffeAlignment(void);
	
	void align_impl(const cv::Mat &img, const cv::Rect& bbox,
            		AlignResult &result);
	private:
		void face_transform(const cv::Mat& img, const LandMarkInfo& src_landmark, 
							cv::Mat& face_img, LandMarkInfo& dst_landmark);
		bool face_transform(const cv::Mat& img, const std::vector<double>& src_landmarks, 
                            cv::Mat& face_img, std::vector<double>& dst_landmarks);
		// FaceHandler _handler;
		FacePara _param;

		FaceAlignWithConf* _cdnn_alignment;
		// CdnnFaceWarp* _face_warpper;

		Cdnn::MPMetricFeature::ExtractMPMetricFeature _transform;

};
}
#endif
