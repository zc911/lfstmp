#ifndef _dgfacesdk_detector_fcn_h_
#define _dgfacesdk_detector_fcn_h_
#include <detector.h>
#include <caffe/caffe.hpp>

#include "FaceDetector.h"
namespace DGFace{

class FcnDetector : public Detector {
    public:
        FcnDetector(int img_scale_max, int img_scale_min,std::string deploy_file, std::string model_file, int gpu_id );
        FcnDetector(int img_scale_max, int img_scale_min, std::string& model_dir, int gpu_id );
        virtual ~FcnDetector(void);
        // detect only -> confidence, bbox
        void detect_impl(const std::vector<cv::Mat> &imgs, std::vector<DetectResult> &results);
    private:

		void ParseConfigFile(std::string cfg_file, std::string& deploy_file, std::string& model_file);

        FCNFaceDetector* _fcn_detecror;
        FacePara _param;
        
};
}
#endif
