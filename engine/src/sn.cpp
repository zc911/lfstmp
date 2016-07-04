//
// Created by chenzhen on 6/30/16.
//
#include <string>
#include "io/ringbuffer.h"
#include "io/stream_tube.h"
#include "vis/display.h"

using namespace std;
using namespace dg;


int main() {
    string video = "/home/chenzhen/video/road1.mp4";

    RingBuffer *buffer = new RingBuffer(100, 640, 480);
    StreamTube *tube = new StreamTube(buffer, video, 25, 1000, 1000, 20000, "TCP");

    tube->StartAsyn();
    Displayer *display = new Displayer(buffer, "aa", 640, 480, 0, 0, 25);
    display->Run();

    while (1) { sleep(10000); }
    return 1;
}