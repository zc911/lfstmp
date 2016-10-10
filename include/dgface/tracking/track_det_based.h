#ifndef _dgfacesdk_tracking_track_det_based_h_
#define _dgfacesdk_tracking_track_det_based_h_

#include <tracking.h>

namespace DGFace{

class Detector;

class DetectionBasedTracking : public Tracking {
    public:
        DetectionBasedTracking(Recognition *recog, Verification *verify, float thresh,
            Detector *detector);
        virtual ~DetectionBasedTracking(void);
    protected:
        // use detector to find objects
        virtual void find_objects(const cv::Mat &img, std::vector<cv::Rect> &bboxes);
    private:
        Detector *_detector;
};
}
#endif

