#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "model/ringbuffer.h"
#include "engine/simple_engine.h"
#include "engine/witness_engine.h"
#include "vis/display.h"

using namespace dg;

int main() {

//    RingBuffer *buffer = new RingBuffer(100);
//    Displayer *displayer = new Displayer(buffer, "Matrix Sample", 1280, 960, 0,
//                                         0, 25);
//    AutoEngine *engine = new SimpleEngine(buffer);
//    StreamTube *tube_ = new StreamTube(buffer, "/home/chenzhen/video/road1.mp4",
//                                       25, 1280, 960, true);
//    tube_->StartAsyn();
//    engine->StartAsyn();
//    displayer->Run();
    Config *config = Config::GetInstance();
    config->Load("config.json");
    SimpleEngine *engine = new WitnessEngine(*config);
    Frame *f = new Frame(1);

    cv::Mat image = imread("test.jpg");
    Payload *payload = new Payload(1, image);
    Operation op;
    op.Set(OPERATION_VEHICLE_DETECT | OPERATION_VEHICLE_STYLE
            | OPERATION_VEHICLE_COLOR | OPERATION_VEHICLE_MARKER
            | OPERATION_VEHICLE_FEATURE_VECTOR);
    f->set_operation(op);
    f->set_payload(payload);

    engine->Process(f);

}
