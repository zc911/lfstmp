#ifndef _DGFACESDK_QUALITY_H_
#define _DGFACESDK_QUALITY_H_

#include <opencv2/opencv.hpp>
#include "common.h"
namespace DGFace{

class Quality {
    public:
        virtual ~Quality(void) {}
        virtual double quality(const cv::Mat &image) = 0;
    protected:
        Quality(void) {}
};

Quality *create_quality(const std::string &prefix = std::string());
}
#endif
