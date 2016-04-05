#ifndef DEBUG_UTIL_H
#define DEBUG_UTIL_H

#include <sys/time.h>
namespace deepglint {

static int TimeCostInMs(struct timeval timeStart, struct timeval timeEnd) {
    return (timeEnd.tv_sec - timeStart.tv_sec) * 1000
            + (timeEnd.tv_usec - timeStart.tv_usec) / 1000;
}

}
#endif
