#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "model/ringbuffer.h"
#include "engine/simple_engine.h"
#include "engine/witness_engine.h"
#include "vis/display.h"
#include "config.h"
using namespace dg;

static void PrintFrame(Frame &frame) {
     cout << "=====FRAME INFO=====" << endl;
     cout << "Frame ID: " << frame.id() << endl;
     Vector<Object *> objs = frame.objects();
     for (int i = 0; i < objs.size(); ++i) {
          Object *obj = objs[i];
          ObjectType type = obj->type();
          if (type >= OBJECT_CAR && type << OBJECT_TRICYCLE) {
               Vehicle *v = (Vehicle*) obj;
               cout << "Vehicle class id: " << v->class_id() << ", Conf: "
                    << v->confidence() << endl;
               cout << "Vehicle color id: " << v->color().class_id << ", "
                    << v->color().confidence << endl;
               cout << "Vehicle plate: " << v->plate().plate_num << ", "
                    << v->plate().confidence << endl;
               vector<Detection> markers = v->markers();
               cout << "Vehicle Markers: " << ", Window: " << v->window().box.x
                    << endl;

               for (int i = 0; i < markers.size(); i++) {
                    Detection d = markers[i];
                    cout << "Marker " << i << ": " << d << endl;
               }
          } else {
               cout << "Type not support now. " << endl;
          }
     }
}

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

     cv::Mat image = cv::imread("test.jpg");
     Payload *payload = new Payload(1, image);
     Operation op;
//    op.Set(OPERATION_VEHICLE_DETECT);
     op.Set(OPERATION_VEHICLE);
     op.Set(OPERATION_VEHICLE_DETECT | OPERATION_VEHICLE_STYLE
               | OPERATION_VEHICLE_COLOR | OPERATION_VEHICLE_MARKER
               | OPERATION_VEHICLE_FEATURE_VECTOR | OPERATION_VEHICLE_PLATE);
     f->set_operation(op);
     f->set_payload(payload);

     FrameBatch *fb = new FrameBatch(1, 1);
     fb->add_frame(f);

     engine->Process(fb);
     PrintFrame(*f);
     DLOG(INFO)<< "FINISHED" << endl;

}
