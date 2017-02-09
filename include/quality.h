#ifndef _DGFACESDK_QUALITY_H_
#define _DGFACESDK_QUALITY_H_

#include <opencv2/opencv.hpp>
#include "common.h"
#include "alignment.h"

namespace DGFace{

enum quality_method {
	BLURM,
	LENET_BLUR,
	FRONT,
	POSE,
};

class Quality {
    public:
        virtual ~Quality(void) {}
        virtual float quality(const cv::Mat &image) {}
        virtual void quality(const std::vector<cv::Mat> &image, std::vector<float> &results) {}
        virtual std::vector<float> quality(const AlignResult &align_result) {}
    protected:
        Quality(void) {}
};

//Quality *create_quality(const std::string &prefix = std::string());
Quality *create_quality(const quality_method& method, const std::string& model_dir, int gpu_id, bool is_encrypt, int batch_size = 1);
Quality *create_quality_with_global_dir(const quality_method& method, const std::string& global_dir,
						int gpu_id = 0,	bool is_encrypt = false, int batch_size = 1);
}
#endif
