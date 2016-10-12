#ifndef _dgfacesdk_detector_rpn_h_
#define _dgfacesdk_detector_rpn_h_
#include <detector.h>
#include <caffe/caffe.hpp>
namespace DGFace{

class RpnDetector : public Detector {
    public:
        RpnDetector(int   img_scale_max,
                    int   img_scale_min,
                    const std::string& model_file,
                    const std::string& trained_file,
                    const std::string& layer_name_cls,
                    const std::string& layer_name_reg,
                    const std::vector <float> area,
                    const std::vector <float> ratio,
                    const std::vector <float> mean,
                    const float  det_thresh  = 0.8,
                    const size_t max_per_img = 100,
                    const size_t stride      = 16,
                    const float  pixel_scale = 1.0f,
                    const bool   use_GPU = true,
                    const int    gpu_id  = 0);
        virtual ~RpnDetector(void);
        // detect only -> confidence, bbox
        void detect_impl(const std::vector<cv::Mat> &imgs, std::vector<DetectResult> &results);
    private:
        void nms(std::vector<Bbox>& p, std::vector<bool>& deleted_mark, float threshold);
        caffe::shared_ptr<caffe::Net<float> > _net;

        std::string _layer_name_cls;
        std::string _layer_name_reg;
        std::vector <float> _area;
        std::vector <float> _ratio;
        std::vector<float> _pixel_means;
        float  _det_thresh;
        size_t _max_per_img;
        size_t _stride;
        float _pixel_scale;
        bool _useGPU;
        int _num_channels;
        int _batch_size;
                int _gpuid;
                bool device_setted_=false;

        cv::Size _image_size;
};
}
#endif
