#ifndef _dgfacesdk_alignment_cdnn_h_
#define _dgfacesdk_alignment_cdnn_h_
#include <string>

#include <alignment.h>
#include "CdnnPosition.h"
#include "ExtractMPMetricFeature.h"
#include "Util.h"
namespace DGFace{
class CdnnAlignment : public Alignment{
	public:
		CdnnAlignment(vector<int> face_size, 
                            string modelDir);
		virtual ~CdnnAlignment(void);
	
	void align_impl(const cv::Mat &img, const cv::Rect& bbox,
            		AlignResult &result);
	private:
		// void face_transform(const cv::Mat& img, const DetectedFaceInfo& det_info, cv::Mat& face_img);
		bool face_transform(const cv::Mat& img, const std::vector<double>& src_landmarks, 
                    cv::Mat& face_img, std::vector<double>& dst_landmarks);
		
		// FaceHandler _handler;
		// FacePara _param;

		Cdnn::sKeyPointsModel _keyPointsModel;
		Cdnn::MPMetricFeature::ExtractMPMetricFeature _transform;

	

};
}
#endif
