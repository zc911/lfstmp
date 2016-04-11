#include <iostream>
#include "model/ringbuffer.h"
#include "engine/simple_engine.h"

using namespace dg;

int main() {

    RingBuffer *buffer = new RingBuffer(100);
    Engine *engine = new SimpleEngine(buffer, NULL);

    engine->StartAsyn();

    while (1) {
        sleep(100);
    }

}
