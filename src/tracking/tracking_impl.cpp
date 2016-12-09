#include <detector.h>
#include <recognition.h>
#include <tracking.h>
#include <tracking/track_det_based.h>
#include <verification.h>

using namespace cv;
using namespace std;

namespace DGFace{
DetectionBasedTracking::DetectionBasedTracking(Recognition *recog, Verification *verify, float thresh, Detector *detector) : Tracking (recog, verify, thresh), _detector(detector) {
}

DetectionBasedTracking::~DetectionBasedTracking(void) {
    delete(_detector);
}

void DetectionBasedTracking::find_objects(const Mat &img, vector<Rect> &bboxes) {
    vector<Mat> imgs;
    vector<DetectResult> results;
    imgs.push_back(img);
    _detector->detect(imgs, results);
    bboxes.clear();
    for (auto &result : results[0].boundingBox) {
        bboxes.push_back(result.second.boundingRect());
    }
}

void DetectionBasedTracking::find_objects(const Mat &img, vector<RotatedRect> &rot_bboxes) {
    vector<Mat> imgs;
    vector<DetectResult> results;
    imgs.push_back(img);
    _detector->detect(imgs, results);
    rot_bboxes.clear();
    for (auto &result : results[0].boundingBox) {
        rot_bboxes.push_back(result.second);
    }
}
/*
Tracking *create_tracker(const std::string &prefix) {
    Config *config = Config::instance();
    string type    = config->GetConfig<string>(prefix + "tracking", "detect_based");
    float thresh   = config->GetConfig<float>(prefix + "tracking.thresh", 0.5f);
    if (type == "detect_based") {
        Recognition *recog   = create_recognition(prefix + "tracking.");
        Verification *verify = create_verifier(prefix + "tracking.");
        Detector *detector   = create_detector(prefix + "tracking.");
        return new DetectionBasedTracking(recog, verify, thresh, detector);
    }
    throw new runtime_error("unknown tracking");
}
*/
Tracking *create_tracker() {
    throw new runtime_error("tracking module will be removed");
}
}
