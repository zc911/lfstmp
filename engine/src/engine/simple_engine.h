/*
 * simple_engine.h
 *
 *  Created on: 11/04/2016
 *      Author: chenzhen
 */

#ifndef SIMPLE_ENGINE_H_
#define SIMPLE_ENGINE_H_

#include "model/model.h"
#include "model/frame.h"
#include "io/stream_tube.h"
#include "processor/simple_processor.h"
#include "processor/vehicle_detector_processor.h"
#include "vis/display.h"

namespace dg {

class SimpleEngine {
 public:

    SimpleEngine() {

    }

    virtual ~SimpleEngine() {

    }

    virtual void Process(Frame *frame) =0;

}
;

}

#endif /* SIMPLE_ENGINE_H_ */
