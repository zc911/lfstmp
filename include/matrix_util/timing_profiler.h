#ifndef TIMEING_PROFILER_H_
#define TIMEING_PROFILER_H_

#include <map>
#include <string>

using namespace std;

namespace dg {

class TimingProfiler {

 public:
    TimingProfiler() {
        cur_time_in_microsend_ = 0;
        time_pieces_.clear();
        history_.clear();
    }
    ~TimingProfiler() {
        time_pieces_.clear();
        history_.clear();
    }

    void Reset(void);
    void Update(string& name);

    float getTimePieceInMillisecend(string& name);
    string getTimeProfileString(void);
    string getSmoothedTimeProfileString(void);

 private:
    unsigned long long cur_time_in_microsend_;

    map<string, float> time_pieces_;
    map<string, float> history_;
};

unsigned long long GetCurrentMicroSecond();
unsigned long long GetCurrentMilliSecond();
}

#endif
