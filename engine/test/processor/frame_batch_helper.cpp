#include <sstream>

#include "frame_batch_helper.h"

using namespace dg;

FrameBatchHelper::FrameBatchHelper(dg::Identification id) {
    frameBatch = new FrameBatch(id);
}

FrameBatchHelper::~FrameBatchHelper() {
    if (frameBatch) {
        delete frameBatch;
        frameBatch = NULL;
    }
}

bool FrameBatchHelper::setImage(const dg::Operation &op,
                                const dg::Identification &id,
                                const string &imgName) {
    cv::Mat image = cv::imread(imgName.c_str());
    if (image.empty()) {
        return false;
    }
    Frame *frame = new Frame(id, image);
    frame->set_operation(op);
    frameBatch->AddFrame(frame);
    return true;
}

int FrameBatchHelper::readImage(const dg::Operation &op) {
    for (int i = 0; ; ++i) {
        stringstream s;
        s << i;
        string imgPath = baseImagePath + string(s.str()) + ".jpg";
        if (setImage(op, i, imgPath) == false) {
            return i;
        }
    }
    return 0;
}

void FrameBatchHelper::printFrame() {
    for (int i = 0; i < frameBatch->batch_size(); ++i) {
        printFrame(frameBatch->frames()[i]);
    }
}

void FrameBatchHelper::printFrame(Frame * frame) {
    cout << "==================FRAME INFO====================" << endl;
    cout << "Frame ID       : " << frame->id() << endl;
    cout << "Frame obj size : " << frame->get_object_size() << endl << endl;

    Vector<Object *> objs = frame->objects();
    for (int i = 0; i < objs.size(); ++i) {
        Object *obj = objs[i];
        ObjectType type = obj->type();
        cout << "---------------------------" << endl;
        cout << "Object type        : " << getType(type) << endl;
        if (type >= OBJECT_CAR && type <= OBJECT_TRICYCLE) {
            Vehicle *v = (Vehicle *) obj;
            cout << "Vehicle class id   : " << v->class_id() << "\t, confidence : "
            << v->confidence() << endl;
            cout << "Vehicle color id   : " << v->color().class_id << "\t, confidence : "
            << v->color().confidence << endl;
            /**
            cout << "Vehicle plate      : " << v->plate().plate_num << ", confidence : "
            << v->plate().confidence << endl;
             **/
            vector<Object *> markers = v->children();
            cout << "Vehicle Markers    : " << v->children().size() << "\t, Window     : "
            << v->window().box.x << endl;

            for (int i = 0; i < markers.size(); i++) {
                Marker *m = (Marker *) markers[i];
                Detection d = m->detection();
                cout << "Marker " << i << ": " << d << " class id: "
                << m->class_id() << endl;
            }

            cout << "Feature Vector     : " << v->feature().Serialize().substr(0, 32)
            << "...    Len : " << v->feature().Serialize().size() << endl;
        }
    }
    cout << endl << endl;
}

string FrameBatchHelper::getType(ObjectType t) {
    string type;
    if (t == OBJECT_FACE) type = "face";
    else if (t == OBJECT_PEOPLE) type = "people";
    else if (t == OBJECT_TRICYCLE) type = "tricycle";
    else if (t == OBJECT_BICYCLE) type = "bicycle";
    else if (t == OBJECT_PEDESTRIAN) type = "pedestrian";
    else if (t == OBJECT_MARKER) type = "marker";
    else if (t == OBJECT_MARKER_0) type = "marker";
    else if (t == OBJECT_MARKER_1) type = "marker";
    else if (t == OBJECT_MARKER_2) type = "marker";
    else if (t == OBJECT_MARKER_3) type = "marker";
    else if (t == OBJECT_MARKER_4) type = "marker";
    else if (t == OBJECT_CAR) type = "car";
    else if (t == OBJECT_UNKNOWN) type = "unknown";
    else type = "Wrong Type!!";

    return type;
}
