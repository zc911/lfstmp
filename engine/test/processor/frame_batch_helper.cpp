#include <sstream>

#include "frame_batch_helper.h"

using namespace dg;

FrameBatchHelper::FrameBatchHelper(dg::Identification id) {
    frameBatch = new FrameBatch(id);
}

FrameBatchHelper::FrameBatchHelper(FrameBatch *fb) {
    frameBatch = fb;
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
        string imgPath = baseImagePath + s.str() + ".jpg";
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
        if (type >= OBJECT_CAR && type <= OBJECT_TRICYCLE && type != OBJECT_PEDESTRIAN) {
            Vehicle *v = (Vehicle *) obj;
            cout << "Vehicle class id   : " << v->class_id() << "\t\t, confidence : "
            << v->confidence() << " " << v->detection().box() << endl;
            cout << "Vehicle color      : " << getVehicleColor(v->color().class_id) << "\t, confidence : "
            << v->color().confidence << endl;
            if (!v->plates().empty()) {
                cout << "Vehicle plate size : " << v->plates().size() << endl;
                for (int i = 0; i < v->plates().size(); ++i) {
                    cout << "Vehicle plate      : " << v->plates()[i].plate_num
                    << ", color id  : " << v->plates()[i].color_id
                    << ", confidence: " << v->plates()[i].confidence << endl;
                }
            }
        /*    vector<Object *> markers = v->children();
            cout << "Vehicle Markers    : " << v->children().size() << "\t\t, Window     : "
            << v->child(OBJECT_WINDOW)->detection().box.x << endl;

            for (int i = 0; i < markers.size(); i++) {
                Marker *m = (Marker *) markers[i];
                Detection d = m->detection();
                cout << "Marker " << i << ": " << d << " class id: "
                << m->class_id() << endl;
            }
*/
            cout << "Feature Vector     : " << v->feature().Serialize().substr(0, 32)
            << "...    Len : " << v->feature().Serialize().size() << endl;
        }
        else if (type == OBJECT_PEDESTRIAN) {
            Pedestrian *v = (Pedestrian*) objs[i];
            cout << "Pedestrian id      : " << v->id() << "\t\t, confidence : "
            << v->confidence() << endl;
            cout << "Attributes size    : " << v->attrs().size() << endl;

            if (!v->attrs().empty()) {
                cout << "---------------------" << endl;
                for (int i = 0; i < v->attrs().size(); ++i) {
                    if (v->attrs()[i].confidence >= 0.5) {
                        cout << v->attrs()[i].tagname << " , "
                        << v->attrs()[i].confidence << endl;
                    }
                }
            }
        }
        else if (type == OBJECT_FACE) {
            Face *f = (Face *)obj;
            cout << "Face id            : " << f->id() << "\t\t, confidence : "
            << f->confidence() << endl;
            cout << "Children size      : " << f->children().size() << endl;
            cout << "Feature Vector     : " << f->feature().Serialize().substr(0, 32)
            << "...    Len : " << f->feature().Serialize().size() << endl;
            cout << f->detection() << endl;
        }
        else {
            cout << "************************" << endl;
            cout << type << endl;
            cout << "************************" << endl;
        }
    }
    cout << endl << endl;
}

string FrameBatchHelper::getType(ObjectType t) {
    string type;
    if (t == OBJECT_FACE) type = "face";
    else if (t == OBJECT_TRICYCLE) type = "tricycle";
    else if (t == OBJECT_BICYCLE) type = "bicycle";
    else if (t == OBJECT_PEDESTRIAN) type = "pedestrian";
    else if (t == OBJECT_MARKER) type = "marker";
    else if (t == OBJECT_CAR) type = "car";
    else if (t == OBJECT_UNKNOWN) type = "unknown";
    else type = "None!!";

    return type;
}

string FrameBatchHelper::getVehicleColor(int t) {
    string color;
    if (t == 0) color = "black";
    else if (t == 1) color = "blue";
    else if (t == 2) color = "brown";
    else if (t == 3) color = "green";
    else if (t == 4) color = "grey";
    else if (t == 5) color = "orange";
    else if (t == 6) color = "pink";
    else if (t == 7) color = "purple";
    else if (t == 8) color = "red";
    else if (t == 9) color = "silver";
    else if (t == 10) color = "white";
    else if (t == 11) color = "yellow";
    else color = "None!!";
    return color;
}
