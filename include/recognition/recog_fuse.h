#ifndef _dgfacesdk_recognition_fuse_h_
#define _dgfacesdk_recognition_fuse_h_
#include <string>
#include <recognition.h>
#include <recognition/recog_cdnn.h>
#include <recognition/recog_cdnn_caffe.h>

namespace DGFace{

class FuseRecog: public Recognition {
	public:
		FuseRecog(std::string config_file, int gpu_id, bool multi_thread, bool is_encrypt, int batch_size);
		virtual ~FuseRecog();
		void recog_impl(const std::vector<cv::Mat>& faces, 
			const std::vector<AlignResult>& alignment,
			std::vector<RecogResult>& results);
	private:
        int cdnn_init(string model_dir, bool multi_thread);
        int cdnn_caffe_init(string model_dir, int gpu_id);
		
		void ParseConfigFile(std::string cfg_file, std::string& cdnn_model_dir, std::string& cdnn_caffe_model_dir, float& cdnn_weight, float& cdnn_caffe_weight);

        void feature_combine(const RecogResult& result_0, const RecogResult& result_1, float wright_0, float weight_1, RecogResult& combined_result);
        void feature_combine(const vector<RecogResult>& results_0, const vector<RecogResult>& results_1, float wright_0, float weight_1, vector<RecogResult>& combined_results);

        Recognition *recog_0, *recog_1;
		float fuse_weight_0, fuse_weight_1;

};
}
#endif
