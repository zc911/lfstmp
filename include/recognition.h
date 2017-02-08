#ifndef _DGFACESDK_DLIB_RECOGNITION_H_
#define _DGFACESDK_DLIB_RECOGNITION_H_

#include "common.h"
#include "alignment.h"
#include "transformation.h"
#include <string>
#include <vector>
namespace DGFace{
enum class recog_method : unsigned char{
	LBP,
	CNN,
	CDNN,
	CDNN_CAFFE,
	FUSION,
	GPU_FUSION,
};

class Recognition {
    public:
        Recognition(bool is_encrypt);
        virtual ~Recognition(void);
        virtual void recog(const std::vector<cv::Mat> &faces, std::vector<RecogResult> &results, const std::string &pre_process);
        virtual void recog(const std::vector<cv::Mat> &faces, const std::vector<AlignResult>& alignment, std::vector<RecogResult> &results, const std::string &pre_process);
    protected:
        virtual void recog_impl(const std::vector<cv::Mat> &faces, std::vector<RecogResult> &results) {}
        virtual void recog_impl(const std::vector<cv::Mat> &faces, const std::vector<AlignResult>& alignment, std::vector<RecogResult> &results) {}

		bool _is_encrypt;
		Transformation* _transformation;
};

//Recognition *create_recognition(const std::string &prefix = std::string());
Recognition *create_recognition(const recog_method& method, const std::string& model_dir, 
								int gpu_id = 0, bool multi_thread = true, 
								bool is_encrypt = false, int batch_size = 1);
Recognition *create_recognition_with_config(const recog_method& method, const std::string& config_file, 
											int gpu_id = 0, bool multi_thread = true,
											bool is_encrypt = false, int batch_size = 1 );
Recognition *create_recognition_with_global_dir(const recog_method& method, const std::string& global_dir, 
											int gpu_id = 0, bool multi_thread = true,
											bool is_encrypt = false, int batch_size = 1 );
}
#endif

