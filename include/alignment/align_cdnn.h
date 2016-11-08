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
		void align_impl(const cv::Mat &img, const cv::RotatedRect& rot_bbox,
						AlignResult &result);
	private:
		void landmarks_transform(const cv::Mat& trans_mat, const std::vector<cv::Point2f>& src_landmarks, std::vector<cv::Point2f>& dst_landmarks);
		
		Cdnn::sKeyPointsModel _keyPointsModel;
};
}
#endif
