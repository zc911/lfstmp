#ifndef _dgfacesdk_quality_blurm_h_
#define _dgfacesdk_quality_blurm_h_

#include <quality.h>
namespace DGFace{

class BlurMQuality : public Quality {
    public:
        BlurMQuality(void);
        virtual ~BlurMQuality(void);
        double quality(const cv::Mat &image);
    private:
        double blur_metric(const cv::Mat &image, short *sobelTable);
};
}
#endif
