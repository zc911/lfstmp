/*
 * vehicle_detector_processor.h
 *
 *  Created on: 13/04/2016
 *      Author: chenzhen
 */

#ifndef VEHICLE_DETECTOR_PROCESSOR_H_
#define VEHICLE_DETECTOR_PROCESSOR_H_

#include <glog/logging.h>
#include "processor.h"
#include "alg/vehicle_multi_type_detector.h"
#include "util/debug_util.h"

namespace dg {

class VehicleMultiTypeDetectorProcessor : public Processor {
 public:
    VehicleMultiTypeDetectorProcessor()
            : Processor() {
        CaffeConfig config;
        config.batch_size = 1;
        config.is_model_encrypt = false;

        if (config.is_model_encrypt) {
            config.model_file =
                    "models/detector/googlenet_faster_rcnn_iter_350000.caffemodel";
            config.deploy_file = "models/detector/test.prototxt";
        } else {
            config.model_file =
                    "models/detector/googlenet_faster_rcnn_iter_350000.caffemodel";
            config.deploy_file = "models/detector/test.prototxt";
        }
        config.use_gpu = true;
        config.gpu_id = 0;
        config.rescale = 400;
        detector_ = new VehicleMultiTypeDetector(config);
    }
    ~VehicleMultiTypeDetectorProcessor() {

    }
    void Update(Frame *frame) {
        DLOG(INFO)<< "Start detect frame: " << frame->id() << endl;

        vector<Detection> detections = detector_->Detect(
                frame->payload()->data());

        for (vector<Detection>::iterator itr = detections.begin();
                itr != detections.end(); ++itr) {
            Detection detection = *itr;
            Object *obj;
            if(d.id == OBJECT_CAR) {
                obj = new Vehicle(OBJECT_CAR);
            } else {
                obj = new Object(d.id);
            }

            obj->set_detection(detection);
            frame->put_object(obj);
            print(d);
        }

        Proceed(frame);

        cout << "End detector frame: " << endl;

    }

    void Update(FrameBatch *frameBatch) {

    }

    bool checkOperation(Frame *frame) {
        return true;
    }
    bool checkStatus(Frame *frame) {
        return true;
    }
private:
    VehicleMultiTypeDetector *detector_;

};

}
#endif /* VEHICLE_DETECTOR_PROCESSOR_H_ */
