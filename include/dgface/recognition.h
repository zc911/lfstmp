#ifndef _DGFACESDK_DLIB_RECOGNITION_H_
#define _DGFACESDK_DLIB_RECOGNITION_H_

#include "common.h"
#include "alignment.h"
#include <string>
#include <vector>
namespace DGFace{

struct RecogResult {
    FeatureType face_feat;
};

class Recognition {
    public:
        Recognition(void);
        virtual ~Recognition(void);
        virtual void recog(const std::vector<cv::Mat> &faces, std::vector<RecogResult> &results, const std::string &pre_process);
        virtual void recog(const std::vector<cv::Mat> &faces, const std::vector<AlignResult>& alignment, std::vector<RecogResult> &results, const std::string &pre_process);
    protected:
        virtual void recog_impl(const std::vector<cv::Mat> &faces, std::vector<RecogResult> &results) {}
        virtual void recog_impl(const std::vector<cv::Mat> &faces, const std::vector<AlignResult>& alignment, std::vector<RecogResult> &results) {}

};
Recognition *create_recognition(const std::string &prefix = std::string());
}
#endif

