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
          DLOG(INFO)<<"Start detect frame: " << frame->id() << endl;
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
          for (int i = 0; i < frameBatch->frames().size(); i++) {
               Frame *frame = frameBatch->frames()[i];
               DLOG(INFO)<< "Start detect frame: " << frame->id() << endl;
               Mat data= frame->payload()->data();
               vector<Detection> detections = detector_->Detect(data);
               int id=0;
               for (vector<Detection>::iterator itr = detections.begin();
                         itr != detections.end(); ++itr) {
                    Detection detection = *itr;
                    Object *obj;
                    if(1) {
                         Vehicle *v = new Vehicle(OBJECT_CAR);
                         obj = static_cast<Object*>(v);
                         obj->set_id(id++);
                         Mat roi = Mat(data, detection.box);
                         v->set_image(roi);
                    }

                    obj->set_detection(detection);
                    frame->put_object(obj);
                    print(detection);
               }
               DLOG(INFO)<<frame->objects().size()<<" "<<detections.size()<<" cars are detected in frame "<<frame->id()<<endl;
               cout << "End detector frame: " << endl;
          }
          Proceed(frameBatch);

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
