//
// Created by chenzhen on 7/6/16.
//

#ifndef PROJECT_SKEYNET_ENGINE_H
#define PROJECT_SKEYNET_ENGINE_H

#include <iostream>
#include <thread>
#include "config.h"
#include "simple_engine.h"
#include "io/stream_tube.h"
#include "io/ringbuffer.h"
#include "vis/display.h"
//#include "alg/detector/detector.h"
#include "processor/processor.h"
#include "processor/config_filter.h"
#include "processor/vehicle_multi_type_detector_processor.h"
#include "processor/parallel/parallel_node.h"
#include "processor/parallel/basic_parallel_processor.h"

#include "model/alg_config.h"

using namespace std;
namespace dg {

class SkynetEngine: public SimpleEngine {
public:
    SkynetEngine(const Config &config) {
        init(config);
    }

    void Run() {
        tube_->StartAsyn();
        std::thread t(&SkynetEngine::process, this);
        display_->Run();
        t.join();
    }

    void process() {
        int current = 0;
        BasicParallelProcessor pp(processor_);
        while (true) {
            Frame *f = buffer_->TryNextFrame(current);

            if (f->CheckStatus(FRAME_STATUS_NEW)) {
                cout << "Process next frame: " << f->id() << " " << f->status() << endl;
                pp.Put(f);
                buffer_->NextFrame(current);

            }
//            usleep(30 * 1000);

        }
    }

    void AsynRun() {

    }

private:


    void init(const Config &config) {
        string video = "/home/chenzhen/video/road1.mp4";
        buffer_ = new RingBuffer(100, 640, 480);
        tube_ = new StreamTube(buffer_, video, 25, 1000, 1000, 20000, "TCP");
        display_ = new Displayer(buffer_, "aa", 640, 480, 0, 0, 25);

        ConfigFilter *configFilter = ConfigFilter::GetInstance();
        configFilter->initDataConfig(config);
        VehicleCaffeDetectorConfig dConfig;
        string dataPath = (string) config.Value("DataPath");
        configFilter->createVehicleCaffeDetectorConfig(config, dConfig);
        processor_ = new VehicleMultiTypeDetectorProcessor(dConfig, false);

    }

    RingBuffer *buffer_;
    StreamTube *tube_;
    Displayer *display_;
    Processor *processor_;

};

}

#endif //PROJECT_SKEYNET_ENGINE_H
