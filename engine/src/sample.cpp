#include <stdio.h>
#include <pthread.h>
#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "model/ringbuffer.h"
#include "engine/simple_engine.h"
#include "engine/witness_engine.h"
#include "vis/display.h"
#include "config.h"
using namespace dg;

// int main(int argc, char **argv)
// {
// 	return 0;
// }

static void PrintFrame(Frame &frame) {
    cout << "=====FRAME INFO=====" << endl;
    cout << "Frame ID: " << frame.id() << endl;
    Vector<Object *> objs = frame.objects();
    for (int i = 0; i < objs.size(); ++i) {
        Object *obj = objs[i];
        ObjectType type = obj->type();
        cout << endl;
        if (type >= OBJECT_CAR && type <= OBJECT_TRICYCLE) {
            cout << "=0^0~ =0^0~ =0^0~ =0^0~ =0^0~ =0^0~ =0^0~ =0^0~ =0^0~ "
                 << endl;
            Vehicle *v = (Vehicle*) obj;
            cout << "Vehicle class id: " << v->class_id() << ", Conf: "
                 << v->confidence() << endl;
            cout << "Vehicle color id: " << v->color().class_id << ", "
                 << v->color().confidence << endl;
            cout << "Vehicle plate: " << v->plate().plate_num << ", "
                 << v->plate().confidence << endl;
            vector<Object *> markers = v->children();
            cout << "Vehicle Markers: " << v->children().size() << ", Window: "
                 << v->window().box.x << endl;

            for (int i = 0; i < markers.size(); i++) {
                Marker *m = (Marker*) markers[i];
                Detection d = m->detection();
                cout << "Marker " << i << ": " << d << " class id: "
                     << m->class_id() << endl;
            }

            cout << "Feature Vector: " << v->feature().Serialize().substr(0, 32)
                 << "... Len: " << v->feature().Serialize().size() << endl;
        } else if (type == OBJECT_FACE) {
            cout << "=.= =.= =.= =.= =.= =.= =.= =.= =.= =.= =.= =.=" << endl;
            Face *f = (Face*) obj;
            cout << "Face Detection: " << f->detection() << endl;
            cout << "Face Vector: " << f->feature().Serialize().substr(0, 32)
                 << "... Len:" << f->feature().Serialize().size() << endl;
        }
    }
}

static void PrintFrame(FrameBatch &frameBatch) {
    for (int i = 0; i < frameBatch.batch_size(); ++i) {
        PrintFrame(*(frameBatch.frames()[i]));
    }
}

static Config *config;
static SimpleEngine *engine1;
//static SimpleEngine *engine2;
//static SimpleEngine *engine3;
//static SimpleEngine *engine4;
//static SimpleEngine *engine5;
//static SimpleEngine *engine6;

static void* process(void* p) {
    SimpleEngine *engine = (SimpleEngine*) p;
    if (1) {
        FrameBatch *fb = new FrameBatch(1111, 2);
        for (int i = 0; i < 1; ++i) {

            char index[1];
            index[0] = '0' + i;
            //   string file = "faces" + string(index) + ".jpg";
            string file = "test.jpg";

            cv::Mat image = cv::imread(file.c_str());

            if (image.empty()) {
                cout << "Read image file failed: " << file << endl;
                return 0;
            }

            Frame *f = new Frame((i + 1) * 100, image);
            Operation op;
            op.Set(OPERATION_VEHICLE);
            op.Set(OPERATION_VEHICLE_DETECT | OPERATION_VEHICLE_STYLE
                    | OPERATION_VEHICLE_COLOR | OPERATION_VEHICLE_MARKER
                    | OPERATION_VEHICLE_FEATURE_VECTOR
                    | OPERATION_VEHICLE_PLATE);
            op.Set(OPERATION_FACE | OPERATION_FACE_DETECTOR
                    | OPERATION_FACE_FEATURE_VECTOR);

            f->set_operation(op);
            fb->add_frame(f);

        }
        engine->Process(fb);
        PrintFrame(*fb);
        delete fb;
    }
    return NULL;
}

int main() {
    config = new Config();
    config->Load("config.json");

    engine1 = new WitnessEngine(*config);
//    engine2 = new WitnessEngine(*config);
//    engine3 = new WitnessEngine(*config);
//    engine4 = new WitnessEngine(*config);
//    engine5 = new WitnessEngine(*config);
//    engine6 = new WitnessEngine(*config);

    pthread_t t1, t2, t3, t4, t5, t6;
    pthread_create(&t1, NULL, process, (void*) engine1);
    //sleep(1);
    // pthread_create(&t2, NULL, process, (void*) engine2);
//    sleep(1);
//    pthread_create(&t3, NULL, process, (void*) engine3);
//    sleep(1);
//    pthread_create(&t4, NULL, process, (void*) engine4);
//    sleep(1);
//    pthread_create(&t5, NULL, process, (void*) engine5);
//    sleep(1);
//    pthread_create(&t6, NULL, process, (void*) engine6);

    while (1) {
        sleep(1111111);
    }
//    for (;;) {
//        FrameBatch *fb = new FrameBatch(1111, 4);
//        for (int i = 0; i < 4; ++i) {
//
//            char index[1];
//            index[0] = '0' + i;
//            string file = "test" + string(index) + ".jpg";
//            cv::Mat image = cv::imread(file.c_str());
//            Frame *f = new Frame((i + 1) * 100, image);
//            Operation op;
//            op.Set(OPERATION_VEHICLE);
//            op.Set(OPERATION_VEHICLE_DETECT | OPERATION_VEHICLE_STYLE
//                    | OPERATION_VEHICLE_COLOR | OPERATION_VEHICLE_MARKER
//                    | OPERATION_VEHICLE_FEATURE_VECTOR
//                    | OPERATION_VEHICLE_PLATE);
//            f->set_operation(op);
//            fb->add_frame(f);
//        }
//
//        engine->Process(fb);
//        PrintFrame(*fb);
//        delete fb;
//    }

    DLOG(INFO)<< "FINISHED" << endl;

}

