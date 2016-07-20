/**
 *     File Name:  frame_batch_helper.h
 *    Created on:  07/18/2016
 *        Author:  Xiaodong Sun
 */

#ifndef TEST_FRAME_HELPER_H_
#define TEST_FRAME_HELPER_H_

#include "processor/processor.h"
#include "processor/vehicle_multi_type_detector_processor.h"


class FrameBatchHelper {

public:
    FrameBatchHelper(dg::Identification id);
    ~FrameBatchHelper();

    bool setImage(const dg::Operation & op,
                  const dg::Identification & id,
                  const string & imgName);

    int readImage(const dg::Operation & op);

    void printFrame();
    void printFrame(dg::Frame *frame);

    string getType(dg::ObjectType t);

    void setBasePath(const string & path) {
        baseImagePath = path;
    }

    dg::FrameBatch* getFrameBatch() {
        return frameBatch;
    }

private:
    dg::FrameBatch *frameBatch;
    string baseImagePath;
};

#endif
