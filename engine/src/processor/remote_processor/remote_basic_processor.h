//
// Created by chenzhen on 7/27/16.
//

#ifndef PROJECT_REMOTE_BASIC_PROCESSOR_H
#define PROJECT_REMOTE_BASIC_PROCESSOR_H

#include <string>
#include "processor.h"

using namespace std;

/// This class is the basic remote processor which defines and implements
/// some basic operations and properties

// TODO Implements
namespace dg {

class RemoteBasicProcessor: public Processor {
public:

    RemoteBasicProcessor();
    ~RemoteBasicProcessor();
    virtual bool process(Frame *frame) = 0;
    virtual bool process(FrameBatch *frame) = 0;

protected:
    string remote_addr_;
};

}
#endif //PROJECT_REMOTE_BASIC_PROCESSOR_H
