/*
 * vehicle_color_processor.h
 *
 *  Created on: Apr 26, 2016
 *      Author: jiajaichen
 */

#ifndef SRC_PROCESSOR_VEHICLE_COLOR_PROCESSOR_H_
#define SRC_PROCESSOR_VEHICLE_COLOR_PROCESSOR_H_



#include "processor/processor.h"
#include "alg/vehicle_caffe_classifier.h"

namespace dg {

class VehicleColorProcessor : public Processor {
 public:
     VehicleColorProcessor() {
        CaffeConfig config;
        config.model_file =
                "models/color/zf_q_iter_70000.caffemodel";
        config.deploy_file = "models/color/deploy.prototxt";

        config.is_model_encrypt = false;
        config.batch_size = 1;
        classifier_ = new VehicleCaffeClassifier(config);
    }

    ~VehicleColorProcessor() {
        if (classifier_)
            delete classifier_;
    }

    virtual void Update(Frame *frame) {

    }

    virtual void Update(FrameBatch *frameBatch) {
         DLOG(INFO)<<"Start detect frame: "<< endl;
         vector<Mat> images = this->vehicles_resized_mat(frameBatch);


         vector<vector<Prediction> > result = classifier_->ClassifyAutoBatch(
                 images);

         SortPrediction(result);

         for(int i=0;i<objs_.size();i++){
              Vehicle *v = (Vehicle*) objs_[i];
              Vehicle::Color color;
              if(result[i].size()<0){
                   continue;
              }
              color.class_id=result[i][0].first;
              color.confidence=result[i][0].second;
              v->set_color(color);
         }
         Proceed(frameBatch);


    }

    virtual bool checkOperation(Frame *frame) {
        return true;
    }
    virtual bool checkStatus(Frame *frame) {
        return true;
    }
 protected:
    vector<Mat > vehicles_resized_mat(FrameBatch *frameBatch) {
          vector<cv::Mat> vehicleMat;
          objs_ = frameBatch->objects();
          for (vector<Object *>::iterator itr = objs_.begin();
                    itr != objs_.end(); ++itr) {
               Object *obj = *itr;

               if (obj->type() == OBJECT_CAR) {

                    Vehicle *v = (Vehicle*) obj;

                    DLOG(INFO)<< "Put vehicle images to be classified: " << obj->id() << endl;
                    vehicleMat.push_back(v->resized_image());

               } else {
                    delete obj;
                    itr = objs_.erase(itr);
                    DLOG(INFO)<< "This is not a type of vehicle: " << obj->id() << endl;
               }
          }
          return vehicleMat;
     }
 private:
    VehicleCaffeClassifier *classifier_;
    vector<Object *>  objs_;
};

}


#endif /* SRC_PROCESSOR_VEHICLE_COLOR_PROCESSOR_H_ */
