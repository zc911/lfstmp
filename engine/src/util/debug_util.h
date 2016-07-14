/*
 * debug_util.h
 *
 *  Created on: 13/04/2016
 *      Author: chenzhen
 */

#ifndef DEBUG_UTIL_H_
#define DEBUG_UTIL_H_

#include <string>
#include <iostream>
#include "model/model.h"

using namespace std;

namespace dg {

static void print(Detection &d) {
    cout << "Detection: " << "[" << d.box.x << "," << d.box.y << ","
        << d.box.width << "," << d.box.height << "] ";
    string type;
    switch (d.id) {
        case 0:
            type = "bg";
            break;
        case 1:
            type = "car";
            break;
        case 2:
            type = "person";
            break;
        case 3:
            type = "bike";
            break;
        case 4:
            type = "tricycle";
            break;

        default:
            type = "Unknown";
            break;
    }
    cout << "Type: " << type << " Conf:" << d.confidence << endl;
}

}

#endif /* DEBUG_UTIL_H_ */
