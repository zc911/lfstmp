/**
 *     File Name:  frame_batch_helper.h
 *    Created on:  07/18/2016
 *        Author:  Xiaodong Sun
 */

#ifndef TEST_FRAME_HELPER_H_
#define TEST_FRAME_HELPER_H_

#include "processor/processor.h"
#include "processor/vehicle_multi_type_detector_processor.h"


class FrameBatchHelper {

public:
    FrameBatchHelper(dg::Identification id);
    ~FrameBatchHelper();

    bool setImage(const dg::Operation & op,
                  const dg::Identification & id,
                  const string & imgName);

    int readImage(const dg::Operation & op);

    void printFrame();
    void printFrame(dg::Frame *frame);

    string getType(dg::ObjectType t);
    string getVehicleColor(int t);

    void setBasePath(const string & path) {
        baseImagePath = path;
    }

    dg::FrameBatch* getFrameBatch() {
        return frameBatch;
    }

    void select() {
        bool flag[100] = {false};
        for (int i = 0; i < frameBatch->batch_size(); ++i) {
            if (frameBatch->frames()[i]->get_object_size() == 1) {
                dg::Object *obj = frameBatch->frames()[i]->objects()[0];
                dg::Vehicle *v = (dg::Vehicle *) obj;
                int id = v->color().class_id;
                cout << "id = " << id << " ";
                if (flag[id] == false) {
                    cout << getVehicleColor(id) << " - - " ;
                    cout << frameBatch->frames()[i]->id() << endl;
                    flag[id] = true;
                }
            }

        }
    }

private:
    dg::FrameBatch *frameBatch;
    string baseImagePath;
};

#endif
