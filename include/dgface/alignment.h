#ifndef _DGFACESDK_ALIGNMENT_H_
#define _DGFACESDK_ALIGNMENT_H_

#include "common.h"
namespace DGFace{

struct AlignResult {
    std::vector<cv::Point>  landmarks;
    cv::Mat                 face_image;
    cv::Rect                bbox;
};

class Alignment{
    public:
        virtual ~Alignment(void);
        void align(const cv::Mat &img, const cv::Rect &bbox, AlignResult &result,
            bool adjust = true);
        void set_avgface(const cv::Mat &img, const cv::Rect &bbox);
    protected:
        std::vector<cv::Point> _avg_points;
        Alignment(std::vector<int> face_size);
        // find landmark only -> landmarks
        virtual void align_impl(const cv::Mat &img, const cv::Rect& bbox, AlignResult &result) = 0; 
    private:
        std::vector<int> _face_size;
};


Alignment *create_alignment(const std::string &prefix = std::string());
}
#endif
