#ifndef _dgfacesdk_quality_frontalm_h_
#define _dgfacesdk_quality_frontalm_h_

#include <quality.h>
#include <detector/det_dlib.h>

namespace DGFace{
class FrontalMQuality : public Quality {
    public:
        FrontalMQuality(void);
        virtual ~FrontalMQuality(void);
        float quality(const cv::Mat &image);
    private:
        Detector* _detector;
};
}
#endif
