/*
 * fs_util.h
 *
 *  Created on: 22/03/2016
 *      Author: chenzhen
 */

#ifndef FS_UTIL_H_
#define FS_UTIL_H_

#include <stdio.h>
#include <string>
#include <cstdio>

using namespace std;
namespace dg {

static int WriteToFile(string filePath, char* data, unsigned int size) {
    FILE* file = fopen(filePath.c_str(), "wr");
    if (file == NULL) {
        return -1;
    }

    fwrite(data, sizeof(char), size, file);
    fflush(file);
    fclose(file);
    return 1;
}

// TODO
static bool CheckExists(string filepath) {
    return false;
}

}
#endif /* FS_UTIL_H_ */
