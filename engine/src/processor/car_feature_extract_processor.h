/*
 * car_feature_extract_processor.h
 *
 *  Created on: Apr 27, 2016
 *      Author: chenzhen
 */

#ifndef CAR_FEATURE_EXTRACT_PROCESSOR_H_
#define CAR_FEATURE_EXTRACT_PROCESSOR_H_

#include "alg/car_feature_extractor.h"
#include "model/frame.h"
#include "model/model.h"
#include "processor.h"

namespace dg {

class CarFeatureExtractProcessor : public Processor {
 public:
    CarFeatureExtractProcessor();
    virtual ~CarFeatureExtractProcessor();

    virtual void Update(Frame *frame);
    virtual void Update(FrameBatch *frameBatch);

    virtual bool checkOperation(Frame *frame) {
        return true;
    }
    virtual bool checkStatus(Frame *frame) {
        return true;
    }
    virtual void beforeUpdate(FrameBatch *frameBatch);
 private:
    CarFeatureExtractor *extractor_;
    void extract(vector<Object*> &objs);

    vector<Object*> vehicle_to_processed_;
};

} /* namespace dg */

#endif /* CAR_FEATURE_EXTRACT_PROCESSOR_H_ */
