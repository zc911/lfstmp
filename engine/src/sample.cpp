#include <iostream>
#include "model/ringbuffer.h"
#include "engine/simple_engine.h"
#include "vis/display.h"

using namespace dg;

int main() {

    RingBuffer *buffer = new RingBuffer(100);
    Displayer *displayer = new Displayer(buffer, "Matrix Sample", 1280, 960, 0,
                                         0, 25);
    AutoEngine *engine = new SimpleEngine(buffer);
    StreamTube *tube_ = new StreamTube(buffer, "/home/chenzhen/video/road1.mp4",
                                       25, 1280, 960, true);
    tube_->StartAsyn();
    engine->StartAsyn();
    displayer->Run();

}
