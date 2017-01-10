/*
 * basic.h
 *
 *  Created on: 29/03/2016
 *      Author: chenzhen
 */

#ifndef BASIC_H_
#define BASIC_H_

#include <stdint.h>
#include <vector>
#include <utility>
#include <opencv2/core/core.hpp>
#include "fs_util.h"
#include "watchdog/watch_dog.h"
using namespace std;

namespace dg {

typedef int64_t Identification;
typedef float Confidence;
typedef int64_t Timestamp;
typedef pair<int, float> Prediction;
typedef cv::Rect Box;

enum ContentType {
    IMAGE_JPEG = 1,
    FILE_MP4 = 2,
    FILE_AVI = FILE_MP4,
    STREAM_RTSP = 4
};

enum MessageStatus {
    MESSAGE_STATUS_INIT = 1,
    MESSAGE_STATUS_SENT = 2,
};

typedef uint64_t OperationValue;
enum Operations
    : OperationValue {
    OPERATION_NONE = 0,
    OPERATION_VEHICLE = 1 << 0,
    OPERATION_VEHICLE_DETECT = 1 << 1,
    OPERATION_VEHICLE_TRACK = 1 << 2,
    OPERATION_VEHICLE_STYLE = 1 << 3,
    OPERATION_VEHICLE_COLOR = 1 << 4,
    OPERATION_VEHICLE_MARKER = 1 << 5,
    OPERATION_VEHICLE_PLATE = 1 << 6,
    OPERATION_VEHICLE_FEATURE_VECTOR = 1 << 7,
    OPERATION_PEDESTRIAN_ATTR = 1 << 8,
    OPERATION_FACE = 1 << 9,
    OPERATION_FACE_DETECT = 1 << 10,
    OPERATION_FACE_FEATURE_VECTOR = 1 << 11,
    OPERATION_DRIVER_BELT = 1 << 12,
    OPERATION_CODRIVER_BELT = 1 << 13,
    OPERATION_DRIVER_PHONE = 1 << 14,
    OPERATION_NON_VEHICLE_ATTR = 1 << 15,
    OPERATION_MAX = 1 << 63
};

class ModelsMap {
public:

    static ModelsMap *GetInstance() {
        static ModelsMap instance;
        return &instance;
    }
    int getModelContent(string file, string & content) {
        map<string, string>::iterator itr = models_.find(file);
        if (itr != models_.end()) {
            content = models_[file];
            return 1 ;

        }
#if DEBUG

        content = ReadStringFromFile(file, "rb");

#else
        int length;
        length = FileSize(file);
        FILE *fp = fopen(file.c_str(), "rb");
        if (fp == NULL) {
            return -1;
        }
        unsigned char *src = (unsigned char *)calloc(length, 1);

        length = fread(src, sizeof(char), length, fp);
        fclose(fp);
        unsigned char * dst = (unsigned char *)calloc(length, 1);
        DecryptModel((unsigned char *)src, length, dst);
        if (file.substr (file.size() - 3, 3) == "txt") {
            string content_tmp((char *)dst);
            content = content_tmp;
        } else {
            string content_tmp((char *)dst, length);
            content = content_tmp;
        }

        free(src);
        free(dst);
#endif
        models_.insert(std::pair<string, string>(file, content));

        return 0;
    }
    void clearModels() {
        models_.clear();
    }
private:
    ModelsMap() {};
    map<string, string> models_;
};

/**
 * This class defines the operations each request(Frame/FrameBatch) asked for.
 * The operation value can be OPERATION_VEHICLE_DETECT or anyone defined in enum OperationValue .
 *
 */
typedef struct Operation {
    OperationValue operate;

    Operation()
        : operate(OPERATION_NONE) {

    }

    bool Check(Operations op) {
        return (operate & op);
    }

    bool Check(OperationValue opv) {
        return (operate & opv);
    }

    void Set(Operations op) {
        operate = (operate | op);
        if (op >= OPERATION_VEHICLE_DETECT
                && op <= OPERATION_VEHICLE_FEATURE_VECTOR) {
            Set(OPERATION_VEHICLE);
        }
        if (op >= OPERATION_FACE_DETECT
                && op <= OPERATION_FACE_FEATURE_VECTOR) {
            Set(OPERATION_FACE);
        }
    }

    void Set(OperationValue opv) {
        operate = (operate | opv);
        if (opv >= OPERATION_VEHICLE_DETECT
                && opv <= OPERATION_VEHICLE_FEATURE_VECTOR) {
            Set(OPERATION_VEHICLE);
        }
        if (opv >= OPERATION_FACE_DETECT
                && opv <= OPERATION_FACE_FEATURE_VECTOR) {
            Set(OPERATION_FACE);
        }
    }

} Operation;

typedef struct {
    int id;
    string uri;
    ContentType type;
    unsigned int width;
    unsigned int height;
    unsigned int chanel;
} MetaData;

typedef struct VideoMetaData: public MetaData {
    unsigned int fps;
} VideoMetaData;

}

#endif /* BASIC_H_ */
