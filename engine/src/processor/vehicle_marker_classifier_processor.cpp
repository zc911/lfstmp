/*
 * vehicle_marker_classifier_processor.cpp
 *
 *  Created on: Apr 27, 2016
 *      Author: jiajaichen
 */

//#include <alg/classification/marker_caffe_classifier.h>
#include "vehicle_marker_classifier_processor.h"
#include "processor_helper.h"
#include "string_util.h"
#include "util/convert_util.h"

using namespace dgvehicle;
namespace dg {

VehicleMarkerClassifierProcessor::VehicleMarkerClassifierProcessor(bool flag)
    : Processor() {

    ssd_marker_detector_ = AlgorithmFactory::GetInstance()->CreateMarkerCaffeSsdDetector();

//    marker_target_min_ = mConfig.target_min_size;
//    marker_target_max_ = mConfig.target_max_size;
    isVisualization_ = flag;
    color_.push_back(Scalar(15, 115, 190));
    color_.push_back(Scalar(230, 2, 0));
    color_.push_back(Scalar(0, 255, 0));
    color_.push_back(Scalar(2, 0, 230));
    color_.push_back(Scalar(255, 255, 0));
    color_.push_back(Scalar(2, 235, 235));
    color_.push_back(Scalar(235, 2, 235));
    color_.push_back(Scalar(155, 255, 0));
    color_.push_back(Scalar(255, 155, 0));
    color_.push_back(Scalar(2, 215, 2));
    color_.push_back(Scalar(2, 115, 2));

}
VehicleMarkerClassifierProcessor::~VehicleMarkerClassifierProcessor() {

    if (ssd_marker_detector_) {
        delete ssd_marker_detector_;
    }

    images_.clear();
    draw_images_.clear();
}

bool VehicleMarkerClassifierProcessor::process(FrameBatch *frameBatch) {

    VLOG(VLOG_RUNTIME_DEBUG) << "Start marker and window processor" << frameBatch->id() << endl;
    VLOG(VLOG_SERVICE) << "Start marker processor" << endl;
    vector<vector<dgvehicle::Detection> > preds;
    ssd_marker_detector_->BatchDetect(images_, fobs_, params_, preds);

    for (int i = 0; i < objs_.size(); i++) {

        Window *v = (Window *) objs_[i];
        vector<Detection> markers_cutborad;
        Mat img(v->image());

        if (isVisualization_) {
            for (int j = 0; j < preds[i].size(); j++) {
                Detection d = ConvertDgvehicleDetection(preds[i][j]);

                markers_cutborad.push_back(d);
                if (d.id > color_.size())
                    continue;
                rectangle(draw_images_[i], preds[i][j].box, color_[d.id]);
                int midx = d.box().x + d.box().width / 2 - 4;
                int midy = d.box().y + d.box().height / 2 + 4;
                string id = i2string(d.id);
                cv::putText(draw_images_[i], id, cv::Point(midx, midy), FONT_HERSHEY_COMPLEX_SMALL, 1, color_[d.id]);
            }
            string name = "marker" + to_string(i) + to_string(draw_images_[i].rows) + ".jpg";
            imwrite(name, draw_images_[i]);
        } else {
            for (int j = 0; j < preds[i].size(); j++) {
                if (preds[i][j].id == 7 || preds[i][j].id == 8) {
                    VLOG(VLOG_RUNTIME_DEBUG) << "Omit the belts found by marker classifier" << endl;
                    continue;
                }
                markers_cutborad.push_back(ConvertDgvehicleDetection(preds[i][j]));
            }
        }
        v->set_markers(markers_cutborad);

    }
    objs_.clear();

    VLOG(VLOG_RUNTIME_DEBUG) << "Finish marker and window processor" << frameBatch->id() << endl;
    return true;
}

bool VehicleMarkerClassifierProcessor::beforeUpdate(FrameBatch *frameBatch) {

#if DEBUG
#else
    if (performance_ > RECORD_UNIT) {
        if (!RecordFeaturePerformance()) {
            return false;
        }
    }
#endif
    objs_.clear();
    images_.clear();
    fobs_.clear();
    params_.clear();
    draw_images_.clear();
    params_.resize(6);
    vector<Object *> objs = frameBatch->CollectObjects(OPERATION_VEHICLE_MARKER);
    vector<Object *>::iterator itr = objs.begin();
    while (itr != objs.end()) {
        Object *obj = *itr;
        if (obj->type() == OBJECT_CAR) {
            for (int i = 0; i < obj->children().size(); i++) {
                Object *obj_child = obj->children()[i];
                if (obj_child->type() == OBJECT_WINDOW) {
                    Window *w = (Window *) obj->children()[i];
                    objs_.push_back(w);
                    fobs_.push_back(w->fobbiden());
                    for (int j = 0; j < params_.size(); j++) {
                        params_[j].push_back(w->params()[j]);
                    }
                    images_.push_back(w->image());
                    performance_++;
                }
            }
            draw_images_.push_back(((Vehicle *) obj)->image());
        } else {
            DLOG(INFO) << "This is not a type of vehicle: " << obj->id() << " " << endl;
        }
        ++itr;

    }

    return true;

}

bool VehicleMarkerClassifierProcessor::RecordFeaturePerformance() {

    return RecordPerformance(FEATURE_CAR_MARK, performance_);

}
}
