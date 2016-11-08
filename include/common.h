#ifndef _DGFACESDK_COMMON_H_
#define _DGFACESDK_COMMON_H_

#include <opencv2/opencv.hpp>
#include <vector>
namespace DGFace{

// Detection
typedef std::pair<float, cv::Rect> Bbox;
typedef std::pair<float, cv::RotatedRect> RotatedBbox;
struct DetectResult {
    std::vector<RotatedBbox> boundingBox;
    cv::Size image_size;
    //std::vector<cv::Point>  landmarks;
    //cv::Mat                 face_image;
};

// Alignment
struct AlignResult {
    std::vector<cv::Point2f>  landmarks;
	std::vector<float>		landmark_scores;
    cv::Mat                 face_image;
    cv::Rect                bbox;
    float                   score;
};

// Feature extraction
typedef float FeatureElemType;
typedef std::vector<FeatureElemType> FeatureType;
struct RecogResult {
    FeatureType face_feat;
};

// Tracking
struct TrackedObj {
    size_t   last_seen;
    size_t   obj_id;
    cv::Rect bbox;
    std::list<FeatureType> last_features;
};

// Database
typedef int FaceIdType;

}
#endif
