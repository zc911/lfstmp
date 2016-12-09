#ifndef _DGFACESDK_DETECTOR_H_
#define _DGFACESDK_DETECTOR_H_

#include <opencv2/opencv.hpp>
#include <vector>
#include "common.h"
namespace DGFace{

enum class det_method : unsigned char{
	DLIB,
	SSD,
	RPN,
	FCN,
};

class Detector {
    public:
        virtual ~Detector(void);
        void detect(const std::vector<cv::Mat> &imgs, std::vector<DetectResult> &results);
        //void set_avgface(const cv::Mat &img);
    protected:
        //std::vector<cv::Point> _avg_points;
        //Detector(int face_size);
        Detector(int img_scale_max, int img_scale_min, bool is_encrypt);
        // detect only -> confidence, bbox
        virtual void detect_impl(const std::vector<cv::Mat> &imgs, std::vector<DetectResult> &results) = 0;

        // find landmark only -> landmarks
        //virtual void landmark_impl(const cv::Mat &img, std::vector<DetectResult> &result) = 0; 
		bool _is_encrypt;
    //private:
        int _img_scale_max; //maximum image edge size (the relative short edge)
        int _img_scale_min; //minimum image edge size (the relative long edge)
        cv::Size _max_image_size;

        void edge_complete(std::vector<cv::Mat> &imgs);
};
//Detector *create_detector(const std::string &prefix = std::string());
Detector *create_detector(const det_method& method, const std::string& model_dir, 
						int gpu_id = 0, bool is_encrypt = false, int batch_size = 1);
Detector *create_detector_with_config(const det_method& method, const std::string& config_file,
						int gpu_id = 0,	bool is_encrypt = false, int batch_size = 1);
}

#endif
