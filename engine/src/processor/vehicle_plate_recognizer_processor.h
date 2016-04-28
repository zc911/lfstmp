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

    PlateRecognizerProcessor();

    ~PlateRecognizerProcessor();
    virtual void Update(Frame *frame) {

    }

    virtual void Update(FrameBatch *frameBatch);
    virtual bool checkOperation(Frame *frame);
    virtual bool checkStatus(Frame *frame);

 protected:

    void sharpenImage(const cv::Mat &image, cv::Mat &result);
    vector<Mat> vehicles_mat(FrameBatch *frameBatch);

 private:
    PlateRecognizer *recognizer_;
    vector<Object *> objs_;

};

}

#endif /* SRC_PROCESSOR_VEHICLE_PLATE_RECOGNIZER_PROCESSOR_H_ */
