#ifndef _dgfacesdk_detector_ssd_h_
#define _dgfacesdk_detector_ssd_h_
#include <detector.h>
#include <caffe/caffe.hpp>
namespace DGFace{

class SSDDetector : public Detector {
    public:
			/*
        SSDDetector(int   img_scale_max,
                    int   img_scale_min,
                    const std::string& model_file,
                    const std::string& trained_file,
                    const std::vector <float> mean,
                    const float det_thresh = 0.8,
                    const float pixel_scale = 1.0f,
                    const bool  use_GPU = true,
                    const int    gpu_id = 0);
					*/
		SSDDetector(int img_scale_max,
					int img_scale_min,
					const std::string& model_dir,
					int gpu_id,
					bool is_encrypt,
					int batch_size);

        virtual ~SSDDetector(void);
        // detect only -> confidence, bbox
        void detect_impl(const std::vector<cv::Mat> &imgs, std::vector<DetectResult> &results);
        void detect(const std::vector<cv::Mat> &imgs, std::vector<DetectResult> &results);
    private:
        void detect_impl_kernel(const std::vector<cv::Mat> &imgs, std::vector<DetectResult> &results);
        // void nms(std::vector<Bbox>& p, std::vector<bool>& deleted_mark, float threshold);
		int ParseConfigFile(const std::string& cfg_file, 
							std::string& deploy_file,
							std::string& model_file); 
		
        caffe::shared_ptr<caffe::Net<float> > _net;
        std::vector<float> _pixel_means;
        float  _det_thresh;
        float _pixel_scale;
        bool _useGPU;
        int _num_channels;
        int _batch_size;
        int _gpuid;
                        bool device_setted_=false;
		bool _bbox_shrink;
		bool _use_deploy_input_size;

        cv::Size _image_size;
};
}
#endif
