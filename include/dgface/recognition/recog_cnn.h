#ifndef _dgfacesdk_recognition_cnn_h_
#define _dgfacesdk_recognition_cnn_h_
#include <recognition.h>
#include <caffe/caffe.hpp>
namespace DGFace{

class CNNRecog: public Recognition {
    public:
        CNNRecog(const std::string& model_file,
                 const std::string& trained_file,
                 const std::string& layer_name,
                 const std::vector <float> mean,
                 const float pixel_scale = 256.0f,
                 const bool use_GPU = true,
                 const int gpu_id   = 0);
        virtual ~CNNRecog(void);
        void recog_impl(const std::vector<cv::Mat> &faces, std::vector<RecogResult> &results);
                void recog_impl(const std::vector<cv::Mat>& faces, 
            const std::vector<AlignResult>& alignment,
            std::vector<RecogResult>& results);
    private:
        caffe::shared_ptr<caffe::Net<float> > _net;
        bool _useGPU;
        std::vector<float> _pixel_means;
        float _pixel_scale;
	    int _num_channels;
        int _batch_size;
        std::string _layer_name;
};
}
#endif

