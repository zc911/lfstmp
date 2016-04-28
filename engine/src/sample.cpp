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
            vector<Object *> markers = v->children();
            cout << "Vehicle Markers: "<<v->children().size() << ", Window: " << v->window().box.x
                 << endl;

            for (int i = 0; i < markers.size(); i++) {
                Marker *m = (Marker*) markers[i];
                Detection d = m->detection();
                cout << "Marker " << i << ": " << d<<" class id: "<<m->class_id()<< endl;
            }

           // cout << "Feature Vector: " << v->feature().Serialize() << endl;
        } else {
            cout << "Type not support now. " << endl;
        }
    }
}

static void PrintFrame(FrameBatch &frameBatch) {
    for (int i = 0; i < frameBatch.batch_size(); ++i) {
        PrintFrame(*(frameBatch.frames()[i]));
    }
}

int main() {

    Config *config = Config::GetInstance();
    config->Load("config.json");
    SimpleEngine *engine = new WitnessEngine(*config);
    FrameBatch *fb = new FrameBatch(1111, 2);

    for (int i = 0; i < 2; ++i) {
        Frame *f = new Frame(i * 100);

        cv::Mat image = cv::imread("test.jpg");
        Payload *payload = new Payload(i * 100, image);
        Operation op;
        op.Set(OPERATION_VEHICLE);
        op.Set(OPERATION_VEHICLE_DETECT | OPERATION_VEHICLE_STYLE
                | OPERATION_VEHICLE_COLOR | OPERATION_VEHICLE_MARKER
                | OPERATION_VEHICLE_FEATURE_VECTOR | OPERATION_VEHICLE_PLATE);
        f->set_operation(op);
        f->set_payload(payload);
        fb->add_frame(f);
    }

    engine->Process(fb);
    PrintFrame(*fb);
    DLOG(INFO)<< "FINISHED" << endl;

}
