//
// Created by jiajaichen on 16-9-23.
//

#include "face_dlib_detector.h"
#include "alg/caffe_helper.h"

/*======================= Dlib detector ========================= */
namespace dg {
FaceDlibDetector::FaceDlibDetector(FaceDetectorConfig &config)
    : img_scale_max_(config.img_scale_max), img_scale_min_(config.img_scale_min),
      _detector(dlib::get_frontal_face_detector()) {
        LOG(INFO)<<img_scale_min_<<" "<<img_scale_max_;
}

FaceDlibDetector::~FaceDlibDetector() {
}

int FaceDlibDetector::Detect( std::vector<cv::Mat> &imgs,vector<vector<Detection>> &detections) {
    for (size_t idx = 0; idx < imgs.size(); idx++) {
        Mat img = imgs[idx];

        float resize_ratio = ReScaleImage(img, img_scale_min_,img_scale_max_);
        if (img.channels() == 4)
            cvtColor(img, img, COLOR_BGRA2GRAY);
        else if (img.channels() == 3)
            cvtColor(img, img, COLOR_BGR2GRAY);

        assert(img.channels() == 1);
        dlib::array2d<unsigned char> dlib_img;
        dlib::assign_image(dlib_img, dlib::cv_image < unsigned char > (img));
        // dlib::cv_image<dlib::bgr_pixel> dlib_img(img);
        vector<pair<double, dlib::rectangle> > dets;
        pyramid_up(dlib_img);
        _detector(dlib_img, dets);
        vector<Detection> boxes;
        for (size_t i = 0; i < dets.size(); ++i) {
            Detection entry;
            auto &det = dets[i];
            entry.confidence = det.first; //confidence
            dlib_rect2cv(det.second, entry.box); //bbox
            entry.box.x /= (2*resize_ratio);
            entry.box.y /= (2*resize_ratio);
            entry.box.width /= (2*resize_ratio);
            entry.box.height /= (2*resize_ratio);
            boxes.push_back(entry);
        }
        detections.push_back(boxes);
    }
    return 1;
}
}