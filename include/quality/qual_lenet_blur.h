#ifndef QUAL_LENET_BLUR_H
#define QUAL_LENET_BLUR_H

#include <quality.h>
#include <caffe/caffe.hpp>

namespace DGFace {
class LenetBlurQuality: public Quality {
public:
	LenetBlurQuality(const std::string& model_dir, int gpu_id);
	virtual ~LenetBlurQuality(void);
	float quality(const cv::Mat &image);
	static const int kBatchSize;
private:
	int ParseConfigFile(const std::string& cfg_file, std::string& deploy_file,
			std::string& model_file);
	caffe::shared_ptr<caffe::Net<float> > _net;
	std::vector<float> _pixel_means;
	float _det_thresh;
	float _pixel_scale;
	bool _useGPU;
	int _num_channels;
	int _batch_size;
	int _gpuid;
	cv::Size _image_size;
};

}

#endif
