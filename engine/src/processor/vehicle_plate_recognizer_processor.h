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

    PlateRecognizerProcessor(const PlateRecognizer::PlateConfig &pConfig);

    ~PlateRecognizerProcessor();

 protected:

    virtual bool process(Frame *frame) {
        return false;
    }
    virtual bool process(FrameBatch *frameBatch);
    virtual bool beforeUpdate(Frame *frame) {
        return false;
    }
    virtual bool beforeUpdate(FrameBatch *frameBatch);

 private:
    void sharpenImage(const cv::Mat &image, cv::Mat &result);
    void filterVehicle(FrameBatch *frameBatch);
 private:
    PlateRecognizer *recognizer_;
    vector<Object *> objs_;
    vector<Mat> images_;
    bool enable_sharpen_;

};

}

#endif /* SRC_PROCESSOR_VEHICLE_PLATE_RECOGNIZER_PROCESSOR_H_ */
