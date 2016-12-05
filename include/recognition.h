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
};

class Recognition {
    public:
        Recognition(void);
        virtual ~Recognition(void);
        virtual void recog(const std::vector<cv::Mat> &faces, std::vector<RecogResult> &results, const std::string &pre_process);
        virtual void recog(const std::vector<cv::Mat> &faces, const std::vector<AlignResult>& alignment, std::vector<RecogResult> &results, const std::string &pre_process);
    protected:
        virtual void recog_impl(const std::vector<cv::Mat> &faces, std::vector<RecogResult> &results) {}
        virtual void recog_impl(const std::vector<cv::Mat> &faces, const std::vector<AlignResult>& alignment, std::vector<RecogResult> &results) {}

		Transformation* _transformation;
};

//Recognition *create_recognition(const std::string &prefix = std::string());
Recognition *create_recognition(const recog_method& method, const std::string& model_dir, int gpu_id, bool multi_thread);
}
#endif

