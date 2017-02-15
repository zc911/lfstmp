#ifndef _dgfacesdk_recognition_cnn_h_
#define _dgfacesdk_recognition_cnn_h_
#include <recognition.h>
#include <caffe/caffe.hpp>
namespace DGFace{

class CNNRecog: public Recognition {
    public:
        CNNRecog(const std::string& model_dir, int gpu_id, bool is_encrypt, int batch_size);
        virtual ~CNNRecog(void);
        void recog_impl(const std::vector<cv::Mat>& faces, const std::vector<AlignResult>& alignment, std::vector<RecogResult>& results);
    private:
        void ParseConfigFile(const std::string& cfg_file,
                                   std::string& model_def,
                                   std::string& weight_file);
        caffe::shared_ptr<caffe::Net<float> > _net;
        std::string _layer_name;
        std::vector<float> _pixel_means;
        std::vector<float> _face_size;
        float _pixel_scale;
        int _batch_size;
        int _num_channels;
        bool _use_gpu;
};
}
#endif

