#include <sstream>
#include <iomanip>
#include <sys/time.h>

#include "timing_profiler.h"

using namespace std;
namespace dg {
void TimingProfiler::Reset(void) 
{
    time_pieces_.clear();
    cur_time_in_microsend_ = GetCurrentMicroSecond();
}

void TimingProfiler::Update(string& name) 
{
    // make sure no other time piece with same name exists
    if (time_pieces_.find(name) != time_pieces_.end()) {
        printf("TimingProfiler::Update error: there is already a time piece called %s exists. \n", name.c_str());
        return;
    }

    unsigned long long curr, diff;
    curr = cur_time_in_microsend_;
    cur_time_in_microsend_ = GetCurrentMicroSecond();
    diff = cur_time_in_microsend_ - curr;

    // insert a new value (int millisecond) in the map;
    time_pieces_[name] = float(diff / 1000.f);

    // apply IIR filter
    float IIR_rate = 0.05f;
    if (history_.find(name) != history_.end()) {
        history_[name] = time_pieces_[name];
    } else {
        history_[name] = IIR_rate * time_pieces_[name]
                + (1.0f - IIR_rate) * history_[name];
    }
}

float TimingProfiler::getTimePieceInMillisecend(string& name) 
{
    // make sure this name piece exists
    if (time_pieces_.find(name) == time_pieces_.end()) 
    {
        printf("TimingProfiler::getTimePieceInMillisecend error: cannot find a time piece called %s. \n", name.c_str());
        return -1.0f;
    }

    return time_pieces_[name];
}

string TimingProfiler::getTimeProfileString(void) 
{
    stringstream ss;
    map<string, float>::iterator it;

    ss << std::fixed << setw(3) << setprecision(1);
    for (it = time_pieces_.begin(); it != time_pieces_.end(); it++) {
        // [name]:float [%s]:%3.1f
        ss << "[" << it->first << "]:" << it->second;
    }
    ss << " µs";

    return ss.str();
}

string TimingProfiler::getSmoothedTimeProfileString(void) 
{
    stringstream ss;
    map<string, float>::iterator it;

    ss << std::fixed << setw(3) << setprecision(1);
    for (it = history_.begin(); it != history_.end(); it++) {
        ss << "[" << it->first << "]:" << it->second;
    }
    ss << " ms";

    return ss.str();
}

// get the time in micro-second (µs)
unsigned long long GetCurrentMicroSecond()
{
    struct timeval start;
    gettimeofday(&start, NULL);
    return (start.tv_sec) * 1000000 + start.tv_usec;
}

// get the time in milli-second (ms)
unsigned long long GetCurrentMilliSecond() 
{
    return GetCurrentMicroSecond() / 1000;
}
}
