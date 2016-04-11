#include "util/timing_profiler.h"

#include <iostream>
#include <fstream>
#include <math.h>
#include <vector>
#include <sstream>
#include <sys/time.h>
#include <unistd.h>
#include <string.h>

using namespace std;
namespace dg {
void timing_profiler::reset(void) {
    time_pieces.clear();

    // get the time in micro-second
    struct timeval start;
    gettimeofday(&start, NULL);
    cur_time_in_microsend = (start.tv_sec) * 1000000 + start.tv_usec;

    return;
}

void timing_profiler::update(string& name) {

    // make sure no other time piece with same name exists
    map<string, float>::iterator it;
    it = time_pieces.find(name);
    if (it != time_pieces.end()) {
        printf("timing_profiler::update error: there is already a time piece called %s exists. \n",
               name.c_str());
        return;
    }

    // get the time in micro-second
    unsigned long long tt;
    struct timeval start;
    gettimeofday(&start, NULL);
    tt = (start.tv_sec) * 1000000 + start.tv_usec;

    unsigned long long diff = tt - cur_time_in_microsend;
    cur_time_in_microsend = tt;

    // insert a new value (int millisecond) in the map;
    float time_in_millisecond = float(diff / 1000.f);
    time_pieces[name] = time_in_millisecond;

    // apply IIR filter
    float IIR_rate = 0.05f;
    it = history.find(name);
    if (it != history.end()) {
        history[name] = IIR_rate * time_pieces[name]
                + (1.0f - IIR_rate) * history[name];
    } else {
        history[name] = time_pieces[name];
    }
}

float timing_profiler::getTimePieceInMillisecend(string& name) {
    // make sure this name piece exists
    map<std::string, float>::iterator it;
    it = time_pieces.find(name);
    if (it == time_pieces.end()) {
        printf("timing_profiler::update error: cannot find a time piece called %s. \n",
               name.c_str());
        return -1.0f;
    }

    return time_pieces[name];
}

char* timing_profiler::getTimeProfileString(void) {
    map<string, float>::iterator it;

    memset(profile_string, 0, 10000);
    for (it = time_pieces.begin(); it != time_pieces.end(); it++) {
        sprintf(profile_string, "%s[%s]:%3.1f, ", profile_string,
                it->first.c_str(), it->second);
    }
    sprintf(profile_string, "%s. [us].", profile_string);

    return profile_string;
}

char* timing_profiler::getSmoothedTimeProfileString(void) {
    map<string, float>::iterator it;

    memset(profile_string, 0, 10000);
    for (it = history.begin(); it != history.end(); it++) {
        sprintf(profile_string, "%s[%s]:%3.1f ", profile_string,
                it->first.c_str(), it->second);
    }
    sprintf(profile_string, "%s ms", profile_string);

    return profile_string;
}

unsigned long long GetCurrentMicroSecond() {
    struct timeval start;
    unsigned long long cur_time_in_microsend;
    gettimeofday(&start, NULL);
    cur_time_in_microsend = (start.tv_sec) * 1000 + start.tv_usec / 1000;
    return cur_time_in_microsend;
}
}
