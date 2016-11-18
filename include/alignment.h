#ifndef _DGFACESDK_ALIGNMENT_H_
#define _DGFACESDK_ALIGNMENT_H_

#include "common.h"
namespace DGFace{



class Alignment{
    public:
        virtual ~Alignment(void);
        void align(const cv::Mat &img, const cv::Rect &bbox, AlignResult &result,
            bool adjust = true);
        void align(const cv::Mat &img, const cv::RotatedRect& rot_bbox, AlignResult &result,
            bool adjust = true);
        void set_avgface(const cv::Mat &img, const cv::Rect &bbox);
        bool is_face(float det_score, float landmark_score, float det_thresh);
    protected:
        std::vector<cv::Point> _avg_points;
        Alignment(std::vector<int> face_size);
        // find landmark only -> landmarks
        virtual void align_impl(const cv::Mat &img, const cv::Rect& bbox, AlignResult &result) = 0; 
        virtual void align_impl(const cv::Mat &img, const cv::RotatedRect& rot_bbox, AlignResult &result) = 0; 
    private:
        std::vector<int> _face_size;
};


Alignment *create_alignment(const std::string &prefix = std::string());
Alignment *create_alignment(const std::string& method, const std::string& model_dir, int gpu_id);

}
#endif
