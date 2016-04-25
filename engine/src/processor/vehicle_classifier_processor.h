/*
 * vehicle_classifier_processor.h
 *
 *  Created on: Apr 22, 2016
 *      Author: chenzhen
 */

#ifndef VEHICLE_CLASSIFIER_PROCESSOR_H_
#define VEHICLE_CLASSIFIER_PROCESSOR_H_

#include "processor/processor.h"
#include "alg/vehicle_caffe_classifier.h"

namespace dg {

class VehicleClassifierProcessor : public Processor {
 public:
    VehicleClassifierProcessor() {
        CaffeConfig config;
        config.model_file =
                "models/classify/car_python_mini_alex_256_0_iter_70000.caffemodel";
        config.deploy_file = "models/classify/deploy_256.prototxt";

        config.is_model_encrypt = false;
        config.batch_size = 1;
        classifier_ = new VehicleCaffeClassifier(config);
    }

    ~VehicleClassifierProcessor() {
        if (classifier_)
            delete classifier_;
    }

    virtual void Update(Frame *frame) {

        vector<Mat> images;
        vector<Object*> objects = frame->objects();

        for (int i = 0; i < objects.size(); ++i) {

            Object *obj = objects[i];
            if (obj->type() == OBJECT_CAR) {

                Vehicle *v = (Vehicle*) obj;
                DLOG(INFO)<< "Put vehicle images to be classified: " << obj->id() << endl;
                images.push_back(v->image());

            } else {
                DLOG(INFO)<< "This is not a type of vehicle: " << obj->id() << endl;
            }

        }

        vector<vector<Prediction> > result = classifier_->ClassifyAutoBatch(
                images);

        SortPrediction(result);

        cout << "Classify result: " << endl;
        for (int i = 0; i < 6; ++i) {
            cout << "Class: " << result[0][i].first << " , Conf: "
                 << result[0][i].second << endl;
        }
    }

    virtual void Update(FrameBatch *frameBatch) {

    }

    virtual bool checkOperation(Frame *frame) {
        return true;
    }
    virtual bool checkStatus(Frame *frame) {
        return true;
    }
 private:
    VehicleCaffeClassifier *classifier_;

};

}

#endif /* VEHICLE_CLASSIFIER_PROCESSOR_H_ */
