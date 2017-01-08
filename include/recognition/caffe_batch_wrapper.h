#ifndef __CAFFE_BATCH_WRAPPER_H__
#define __CAFFE_BATCH_WRAPPER_H__

#include <caffe/caffe.hpp>
#include <memory>
#include <string>
#include <vector>

namespace DGFace {
class CaffeBatchWrapper {
	public:
		CaffeBatchWrapper(int deviceId, const std::string &layer_name, int batch_size,
            const std::string &model_def, const std::string &weights,
            const std::vector<int> &patch_ids, bool is_encrypt);
        void predict(const std::vector<cv::Mat> &images,
            const std::vector< std::vector<float> > &landmarks,
            std::vector< std::vector<float> > &output);
        int batch_size(void) {return _batch_size;}
	private:
		void ReadFacePatchImageToData(const cv::Mat &img,
			const std::vector<float> & landmarks, float *transformed_data);
        int    _batch_size;
        size_t _feature_size;
		std::unique_ptr< caffe::Net<float> > _net;
		std::unique_ptr<caffe::NetParameter> _net_param;
	    caffe::Blob<float>* _input_blob;
	    caffe::Blob<float>* _output_blob;
        std::vector<int>    _patch_ids;
};
}
#endif
