/*
 * vehicle_color_processor.h
 *
 *  Created on: Apr 26, 2016
 *      Author: jiajiachen
 */

#ifndef SRC_PROCESSOR_VEHICLE_COLOR_PROCESSOR_H_
#define SRC_PROCESSOR_VEHICLE_COLOR_PROCESSOR_H_

#include "processor/processor.h"
//#include "alg/classification/vehicle_caffe_classifier.h"
#include "processor_helper.h"
//#include "alg/classification/caffe_vehicle_color_classifier.h"
#include "model/basic.h"
#include "algorithm_factory.h"
#include "util/caffe_helper.h"

namespace dg {
static int SHIFT_COLOR = 1000;
class VehicleColorProcessor: public Processor {
public:
    VehicleColorProcessor(bool enable_demo);
    ~VehicleColorProcessor();

protected:
    virtual bool process(Frame *frame) {
        return false;
    }
    virtual bool process(FrameBatch *frameBatch);
    virtual bool beforeUpdate(FrameBatch *frameBatch);


    virtual bool RecordFeaturePerformance();
    void score_color(Prediction &max, vector<Prediction> preds) {
        Prediction min = nthPrediction(preds, 0);
        Prediction fth = nthPrediction(preds, preds.size() - 1);
        Prediction sth = nthPrediction(preds, preds.size() - 2);
        Prediction tth = nthPrediction(preds, preds.size() - 3);
        float tot_score = 0;
        for (int i = 0; i < preds.size(); i++) {
            tot_score += preds[i].second - min.second;
        }
        float high_thr = color_high_thr_ * tot_score + min.second;
        float low_thr = color_low_thr_ * tot_score + min.second;
        max.first = -1;
        max.second = 0;
        if (fth.second > high_thr) {
            max = fth;
            return;
        }
        if (tth.second > low_thr) {
            max.first = -1;
            max.second = 0;
            return;
        }
        if (sth.second > low_thr) {
            int tmp_min = std::min(fth.first, sth.first);
            int tmp_max = std::max(fth.first, sth.first);

            max.first = preds.size() * (tmp_min + 1) - (tmp_min) * (tmp_min + 1) / 2 + tmp_max - tmp_min - 1;
            max.second = 1;
            return;
        }
    }
    void normalize_color(Prediction &max, vector<Prediction> preds) {
        Prediction min = nthPrediction(preds, 0);
        max = nthPrediction(preds, preds.size() - 1);
        float tot_score = 0;

        for (int i = 0; i < preds.size(); i++) {
            tot_score += preds[i].second - min.second;
        }
        max.second = (max.second - min.second) / tot_score;
    }
protected:
    void vehiclesResizedMat(FrameBatch *frameBatch);
private:

    vector<dgvehicle::AlgorithmProcessor *> classifiers_;
    vector<Object *> objs_;
    vector<Mat> images_;
    float color_low_thr_ = 0.3;
    float color_high_thr_ = 0.5;

};

}

#endif /* SRC_PROCESSOR_VEHICLE_COLOR_PROCESSOR_H_ */
