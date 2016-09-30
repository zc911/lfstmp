#ifndef _DGFACESDK_TRACKING_H_
#define _DGFACESDK_TRACKING_H_

#include "common.h"

#include <list>
#include <map>
namespace DGFace{
struct TrackedObj {
    size_t   last_seen;
    size_t   obj_id;
    cv::Rect bbox;
    std::list<FeatureType> last_features;
};

class Recognition;
class Verification;

class Tracking {
    public:
        virtual ~Tracking(void);
        void update(const cv::Mat &img);
        const std::vector<TrackedObj> &get_objects() const { return _objects; }
        const size_t frame_index() const {return _frame_index;}

    protected:
        Tracking(Recognition *recog, Verification *verify, float thresh);
        virtual void find_objects(const cv::Mat &img, std::vector<cv::Rect> &bboxes) = 0;

        size_t           _next_id;
        size_t           _frame_index;
        Recognition     *_recognize;
        Verification    *_verifier;
        float            _verify_thresh;
        std::vector<TrackedObj> _objects;

    private:
        void extract_features(const cv::Mat &img,
            const std::vector<cv::Rect> &bboxes,
            std::vector<FeatureType> &features);
        void build_similarity(const std::vector<cv::Rect> &now_bboxes,
            const std::vector<FeatureType> &now_features,
            std::vector<std::vector<float> > &similarity);
        void match_object(const std::vector<std::vector<float> >&similarity,
            std::vector<ssize_t> &match_result);
        void post_process(const std::vector<ssize_t> &match_result,
            const std::vector<cv::Rect> &now_bboxes,
            const std::vector<FeatureType> &now_features);
        std::vector<cv::Mat>     _image_cache;
        std::vector<cv::Rect>    _bbox_cache;
        std::vector<FeatureType> _feature_cache;
        std::vector<ssize_t>     _match_cache;
        std::vector<bool>        _matched_flags;
        std::vector<std::vector<float> > _similariy_cache;
};

Tracking *create_tracker(const std::string &prefix = std::string());
}
#endif

