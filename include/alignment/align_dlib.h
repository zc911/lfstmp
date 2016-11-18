#ifndef _dgfacesdk_alignment_dlib_h_
#define _dgfacesdk_alignment_dlib_h_
#include <string>
#include "dlib/image_processing.h"
#include <alignment.h>
namespace DGFace{
// Dlib 68 landmarks alignmnet
class DlibAlignment : public Alignment {
    public:
        DlibAlignment(std::vector<int> face_size, const std::string &align_model, const std::string &det_type);
        virtual ~DlibAlignment(void);
        // find landmark only -> landmarks
        void align_impl(const cv::Mat &img, const cv::Rect& bbox,
            AlignResult &result); 
        void align_impl(const cv::Mat &img, const cv::RotatedRect& rot_bbox,
            AlignResult &result); 
    private:
        dlib::shape_predictor _sp;
        std::string _det_type;

};
}
#endif
