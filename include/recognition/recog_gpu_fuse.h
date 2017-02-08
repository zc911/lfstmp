#ifndef _dgfacesdk_recognition_gpu_fuse_h_
#define _dgfacesdk_recognition_gpu_fuse_h_
#include <string>
#include <vector>
#include <recognition.h>
#include <recognition/recog_cdnn_caffe.h>
namespace DGFace {

class GPUFuseRecog: public Recognition {
	public:
		GPUFuseRecog(const std::string& model_dir, int gpu_id, bool is_encrypt, int batch_size);
		virtual ~GPUFuseRecog();
		void recog_impl(const std::vector<cv::Mat>& faces, 
			const std::vector<AlignResult>& alignments,
			std::vector<RecogResult>& results);
	protected:
		void ParseConfigFile(const std::string& cfg_file, std::vector<std::string>& model_dir, 
					std::vector<float>& fuse_weight);
		void feature_combine(const std::vector<RecogResult>& result, RecogResult& combined_result);
		void feature_combine(const std::vector<std::vector<RecogResult> >& results, 
					std::vector<RecogResult>& combined_results);

		std::vector<Recognition*> _recognizers;
		std::vector<float> _fuse_weights;
};
}
#endif
