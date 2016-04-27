/*
 * vehicle_plate_recognizer.h
 *
 *  Created on: Apr 27, 2016
 *      Author: jiajaichen
 */

#ifndef SRC_PROCESSOR_VEHICLE_PLATE_RECOGNIZER_PROCESSOR_H_
#define SRC_PROCESSOR_VEHICLE_PLATE_RECOGNIZER_PROCESSOR_H_


#include "processor/processor.h"
#include "alg/plate_recognizer.h"

namespace dg {

class PlateRecognizerProcessor : public Processor {
 public:
     PlateRecognizerProcessor() {
          PlateRecognizer::PlateConfig pConfig;
          pConfig.LocalProvince = "";

          pConfig.OCR = 1;
          pConfig.PlateLocate = 5;

          recognizer_ = new PlateRecognizer(pConfig);
    }

    ~PlateRecognizerProcessor() {
        if (recognizer_)
            delete recognizer_;
    }

    virtual void Update(Frame *frame) {

    }

    virtual void Update(FrameBatch *frameBatch) {
         DLOG(INFO)<<"Start detect frame: "<< endl;
         vector<Mat> vehicles = this->vehicles_mat(frameBatch);

         for(int i=0;i<vehicles.size();i++) {
              Vehicle *v = (Vehicle*) objs_[i];
              Mat tmp = vehicles[i];
              Vehicle::Plate pred = recognizer_->Recognize(tmp);
              DLOG(INFO)<<"plate number "<<pred.plate_num<<endl;
              v->set_plate(pred);
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
    void sharpenImage(const cv::Mat &image, cv::Mat &result) {
         //创建并初始化滤波模板
         cv::Mat kernel(3, 3, CV_32F, cv::Scalar(0));
         kernel.at<float>(1, 1) = 5.0;
         kernel.at<float>(0, 1) = -1.0;
         kernel.at<float>(1, 0) = -1.0;
         kernel.at<float>(1, 2) = -1.0;
         kernel.at<float>(2, 1) = -1.0;

         result.create(image.size(), image.type());

         //对图像进行滤波
         cv::filter2D(image, result, image.depth(), kernel);
    }
    vector<Mat > vehicles_mat(FrameBatch *frameBatch) {
          vector<cv::Mat> vehicleMat;
          objs_ = frameBatch->objects();
          for (vector<Object *>::iterator itr = objs_.begin();
                    itr != objs_.end(); ++itr) {
               Object *obj = *itr;

               if (obj->type() == OBJECT_CAR) {

                    Vehicle *v = (Vehicle*) obj;

                    DLOG(INFO)<< "Put vehicle images to be classified: " << obj->id() << endl;
                    vehicleMat.push_back(v->image());

               } else {
                    delete obj;
                    itr = objs_.erase(itr);
                    DLOG(INFO)<< "This is not a type of vehicle: " << obj->id() << endl;
               }
          }
          return vehicleMat;
     }
 private:
    PlateRecognizer *recognizer_;
    vector<Object *>  objs_;

};

}



#endif /* SRC_PROCESSOR_VEHICLE_PLATE_RECOGNIZER_PROCESSOR_H_ */
