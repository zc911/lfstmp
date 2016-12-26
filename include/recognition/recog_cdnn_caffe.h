#ifndef _dgfacesdk_recognition_cdnn_caffe_h_
#define _dgfacesdk_recognition_cdnn_caffe_h_
#include <string>
#include <recognition.h>
#include "FaceFeature.h"
namespace DGFace{

class CaffeBatchWrapper;

class CdnnCaffeRecog: public Recognition {
	public:
		CdnnCaffeRecog(const std::string& model_dir, int gpu_id, bool is_encrypt, int batch_size);
		virtual ~CdnnCaffeRecog();
		void recog_impl(const std::vector<cv::Mat>& faces, 
			const std::vector<AlignResult>& alignment,
			std::vector<RecogResult>& results);
	private:

		void ParseConfigFile(const std::string& cfg_content,
								   std::vector<std::string>& model_defs, 
								   std::vector<std::string>& weight_files, 
								   std::vector<std::string>& layer_names,
								   std::vector<int>& patch_ids,
								   std::vector<int>& patch_dims,
									bool* is_color);
        void ProcessBatchAppend(const std::vector<cv::Mat> &images,
                std::vector< std::vector<float> > &landmarks,
                std::vector<RecogResult> &results);

        int _batch_size;
        CaffeBatchWrapper *_impl;

};
}
#endif
