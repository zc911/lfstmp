/**
 *     File Name:  vehicle_processor_head.h
 *    Created on:  07/18/2016
 *        Author:  Xiaodong Sun
 */

#ifndef TEST_VEHICLE_PROCESSOR_HEAD_H_
#define TEST_VEHICLE_PROCESSOR_HEAD_H_

#include "processor/processor.h"
#include "processor/vehicle_multi_type_detector_processor.h"
#include "processor/vehicle_window_detector_processor.h"
class VehicleProcessorHead {

public:
    VehicleProcessorHead();
    ~VehicleProcessorHead();

//    dg::VehicleCaffeDetectorConfig getConfig();
    void init();

    dg::Processor *getProcessor() {
        return processor;
    }

    void setNextProcessor(dg::Processor *p) {
        processor->SetNextProcessor(p);
    }

    void process(dg::FrameBatch *frameBatch) {
        processor->Update(frameBatch);
    }

private:
    dg::VehicleMultiTypeDetectorProcessor *processor;
};
class VehicleWindowProcessor {

public:
    VehicleWindowProcessor();
    ~VehicleWindowProcessor();

//    dg::VehicleCaffeDetectorConfig getConfig();
    void init();

    dg::Processor *getProcessor() {
        return processor;
    }

    void setNextProcessor(dg::Processor *p) {
        processor->SetNextProcessor(p);
    }

    void process(dg::FrameBatch *frameBatch) {
        processor->Update(frameBatch);
    }

private:
    dg::VehicleWindowDetectorProcessor *processor;
};
#endif
