#ifndef _dgfacesdk_recognition_cdnn_caffe_h_
#define _dgfacesdk_recognition_cdnn_caffe_h_
#include <string>
#include <recognition.h>
#include "FaceFeature.h"
#include "feature_extractor.h"
#include "face_inf.h"
namespace DGFace{

class CdnnCaffeRecog: public Recognition {
	public:
		CdnnCaffeRecog(const std::string& model_dir, int gpu_id);
		virtual ~CdnnCaffeRecog();
		void recog_impl(const std::vector<cv::Mat>& faces, 
			const std::vector<AlignResult>& alignment,
			std::vector<RecogResult>& results);
	private:

		void ParseConfigFile(const string& cfg_content,
									vector<string>& model_defs, 
									vector<string>& weight_files, 
									vector<string>& layer_names,
									vector<int>& patch_ids,
									vector<int>& patch_dims,
									bool* is_color);

		FacePara _param;
		bool multi_thread_;
        bool multi_process_;
        void * m_CaffeHandler;

};
}
#endif
