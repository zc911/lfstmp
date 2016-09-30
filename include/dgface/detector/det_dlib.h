#ifndef _dgfacesdk_detector_dlib_h_
#define _dgfacesdk_detector_dlib_h_
#include <detector.h>
#include "../dlib/image_processing/frontal_face_detector.h"
namespace DGFace{

class DlibDetector : public Detector {
    public:
        DlibDetector(int img_scale_max, int img_scale_min);
        virtual ~DlibDetector(void);
        // detect only -> confidence, bbox
        void detect_impl(const std::vector<cv::Mat> &imgs, std::vector<DetectResult> &results);
    private:
        dlib::frontal_face_detector _detector;
};
}
#endif
