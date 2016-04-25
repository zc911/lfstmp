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

const static string MODEL_FILE =
        "models/detector/googlenet_faster_rcnn_iter_350000.caffemodel";
const static string DEPLOY_FILE = "models/detector/test.prototxt";
const static string MODEL_FILE_EN =
        "models/detector/googlenet_faster_rcnn_iter_350000.caffemodel";
const static string DEPLOY_FILE_EN = "models/detector/test.prototxt";

class VehicleMultiTypeDetectorProcessor : public Processor {
 public:

    VehicleMultiTypeDetectorProcessor(int batch_size, bool gpu_id, int rescale,
                                      bool is_model_encrypt)
            : Processor() {

        CaffeConfig config;
        config.batch_size = batch_size;
        config.is_model_encrypt = is_model_encrypt;

        if (config.is_model_encrypt) {
            config.model_file = MODEL_FILE;
            config.deploy_file = DEPLOY_FILE;
        } else {
            config.model_file = MODEL_FILE_EN;
            config.deploy_file = DEPLOY_FILE_EN;
        }

        config.use_gpu = true;
        config.gpu_id = gpu_id;
        config.rescale = rescale;
        detector_ = new VehicleMultiTypeDetector(config);
    }

    ~VehicleMultiTypeDetectorProcessor() {

    }

    void Update(Frame *frame) {

        DLOG(INFO)<< "Start detect frame: " << frame->id() << endl;
        Mat data= frame->payload()->data();
        vector<Detection> detections = detector_->Detect(data);

        for (vector<Detection>::iterator itr = detections.begin();
                itr != detections.end(); ++itr) {
            Detection detection = *itr;
            Object *obj;
            if(1) {
                Vehicle *v = new Vehicle(OBJECT_CAR);
                obj = static_cast<Object*>(v);
                Mat roi = Mat(data, detection.box);
                v->set_image(roi);
            }

            obj->set_detection(detection);
            frame->put_object(obj);
            print(detection);
        }
        cout << "End detector frame: " << endl;
        Proceed(frame);

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
