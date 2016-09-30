#ifndef _dgfacesdk_recognition_lbp_h_
#define _dgfacesdk_recognition_lbp_h_

#include <recognition.h>
namespace DGFace{

class LbpRecog: public Recognition {
    public:
        LbpRecog(int radius, int neighbors, int grid_x, int grid_y);
        void recog_impl(const std::vector<cv::Mat> &faces, std::vector<RecogResult> &results); 
    private:
        int _radius;
        int _neighbors;
        int _grid_x;
        int _grid_y;
};
}
#endif


