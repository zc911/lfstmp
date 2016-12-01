/*
 * simple_engine.h
 *
 *  Created on: 11/04/2016
 *      Author: chenzhen
 */

#ifndef SIMPLE_ENGINE_H_
#define SIMPLE_ENGINE_H_

#include "model/frame.h"

namespace dg {

class SimpleEngine {
public:

    SimpleEngine() {

    }

    virtual ~SimpleEngine() {

    }

    virtual void Process(FrameBatch *frame) { };
    virtual void Run() { };
    virtual void AsynRun() { };

};

}

#endif /* SIMPLE_ENGINE_H_ */
