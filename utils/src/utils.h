/*
 * utils.h
 *
 *  Created on: 24/03/2016
 *      Author: chenzhen
 */

#ifndef UTILS_H_
#define UTILS_H_

#include <stdlib.h>
#include <uuid/uuid.h>
#include <sys/time.h>
namespace dg {
static long long int GenerateUid() {
    long long int num = 0;
    uuid_t uuid;
    uuid_generate_time_safe(uuid);
    for (int i = 0; i < 8; ++i) {
        long long int a = uuid[i];
        long long int b = (a << (7 - i) * 8);
        num = num | b;
    }
    return num;
}

static struct timeval nowTime;
static long long int NowMs() {
    gettimeofday(&nowTime, NULL);
    return nowTime.tv_sec * 1000 + nowTime.tv_usec / 1000;
}
static string GetLatestHour(){
    time_t current_time;
    struct tm *struct_time;
    time(&current_time);
    struct_time=gmtime(&current_time);
    string result = to_string(struct_time->tm_year)+to_string(struct_time.tm_mon)+to_string(struct_time->tm_mday)+to_string(struct_time->tm_hour);
    return result;
}
static long long int NowSeconds() {
    gettimeofday(&nowTime, NULL);
    return nowTime.tv_sec;
}

}
#endif /* UTILS_H_ */
