#ifndef _DGFACESDK_DLIB_RECOGNITION_H_
#define _DGFACESDK_DLIB_RECOGNITION_H_

#include "common.h"
#include <string>
#include <vector>
namespace DGFace{

struct RecogResult {
    FeatureType face_feat;
};

class Recognition {
    public:
        virtual ~Recognition(void);
        void recog(const std::vector<cv::Mat> &faces, std::vector<RecogResult> &results, const std::string &pre_process);
    protected:
        Recognition(void);
        virtual void recog_impl(const std::vector<cv::Mat> &faces, std::vector<RecogResult> &results) = 0;
};

Recognition *create_recognition(const std::string &prefix = std::string());
}
#endif

