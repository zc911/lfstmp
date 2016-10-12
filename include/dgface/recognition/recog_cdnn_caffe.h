#ifndef _dgfacesdk_recognition_cdnn_caffe_h_
#define _dgfacesdk_recognition_cdnn_caffe_h_
#include <string>
#include <recognition.h>
#include "FaceFeature.h"
#include "feature_extractor.h"
namespace DGFace{

class CdnnCaffeRecog: public Recognition {
	public:
		CdnnCaffeRecog(std::string config_file, int gpu_id);
		virtual ~CdnnCaffeRecog();
		void recog_impl(const std::vector<cv::Mat>& faces, 
			const std::vector<AlignResult>& alignment,
			std::vector<RecogResult>& results);
	private:
		// FaceHandler _handler;
		FacePara _param;
		// CaffeFaceFeature* _cdnn_extractor;
		bool multi_thread_;
        bool multi_process_;
        void * m_CaffeHandler;

};
}
#endif
