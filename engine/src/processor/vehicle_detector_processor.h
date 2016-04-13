/*
 * vehicle_detector_processor.h
 *
 *  Created on: 13/04/2016
 *      Author: chenzhen
 */

#ifndef VEHICLE_DETECTOR_PROCESSOR_H_
#define VEHICLE_DETECTOR_PROCESSOR_H_

#include "processor.h"
#include "alg/faster_rcnn_detector.h"
#include "util/debug_util.h"

namespace dg {

class VehicleDetectorProcessor : public Processor {
 public:
    VehicleDetectorProcessor()
            : Processor() {
        CaffeConfig config;
        config.batch_size = 1;
        config.model_file =
                "models/detector/googlenet_faster_rcnn_iter_350000.caffemodel";
        config.deploy_file = "models/detector/train.prototxt";
        config.use_gpu = true;
        config.gpu_id = 0;
        config.rescale = 400;
        detector_ = new FasterRcnnDetector(config);
    }
    ~VehicleDetectorProcessor() {

    }
    void Update(Frame *frame) {
        if (!checkOperation(frame)) {
            return;
        }
        if (!checkStatus(frame)) {
            return;
        }

        vector<Detection> detections = detector_->Detect(
                frame->payload()->data());

        for (vector<Detection>::iterator itr = detections.begin();
                itr != detections.end(); ++itr) {
            Detection d = *itr;
            Object *obj = new Object();
            obj->set_detection(d);
            frame->put_object(obj);
            print(d);
        }

        frame->set_status(FRAME_STATUS_DETECTED);

    }
    void Update(FrameBatch *frameBatch) {

    }

    bool checkOperation(Frame *frame) {
        return true;
    }
    bool checkStatus(Frame *frame) {
        if (frame->status() != FRAME_STATUS_INIT
                || frame->status() == FRAME_STATUS_DETECTED) {
            return false;
        }
        return true;
    }
 private:
    Detector *detector_;

};

}
#endif /* VEHICLE_DETECTOR_PROCESSOR_H_ */
