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
		CdnnCaffeAlignment(std::vector<int> face_size, std::string model_dir, int gpu_id, bool is_encrypt);
		virtual ~CdnnCaffeAlignment(void);
	
		void align_impl(const cv::Mat &img, const cv::Rect& bbox,
						AlignResult &result);
		void align_impl(const cv::Mat &img, const cv::RotatedRect& rot_bbox,
						AlignResult &result);
	private:

		void ParseConfigFile(const std::string& cfg_file, 
							std::vector<std::string>& deploy_files, 
							std::vector<std::string>& model_files);

		FacePara _param;
		FaceAlignWithConf* _cdnn_alignment;

};
}
#endif
