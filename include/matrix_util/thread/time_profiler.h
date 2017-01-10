//
// Created by jiajaichen on 16-7-7.
//

#ifndef PROJECT_TIME_PROFILER_H
#define PROJECT_TIME_PROFILER_H
namespace dg{

class DVProfStats {
public:
    DVProfStats(string prefix)
        : prefix_(prefix), requests(0), images(0)
        , max(0), min(0xfffffffffffffff), cur(0), sum(0)
    {
    }
    virtual ~DVProfStats(){}
    void update(map<string, uint64_t>& m, string end, string start, int imgs)
    {
        if ( m.find(end) == m.end() ) return;
        if ( m.find(start) == m.end() ) return;
        update(m[end] - m[start], imgs);
    }

    void update(int64_t val, int imgs)
    {
        if( imgs == 0 )return;
        requests++;
        images+= imgs;
        cur = val;
        max = (val > max) ? val : max;
        min = (val < min) ? val : min;
        sum += val;
    }

    void dump()
    {
        requests = requests == 0 ? 1 : requests;
        LOG(WARNING) << "[DUMP] " << prefix_ << ": max: " << max << ", min: " << min << ", cur: " << cur << ", avg: " << (sum / requests);
        max = min = cur = sum = 0;
        requests = images = 0;
    }

private:
    string prefix_;
    int requests, images;
    int64_t max, min, cur, sum;
};

class DvProfiler {
public:
    DvProfiler()
        : latest_(gettime())
        , prev_proc_("before_proc")
        , classify_("classify")
        , detect_("detect")
        , color_("color")
        , plate_("plate")
        , marker_("marker")
        , feature_("feature")
        , next_proc_("after_proc")
        , total_("total")
        , requests_(0)
        , images_(0)
        , fails_(0)
        , fimgs_(0)
    {
    }
    virtual ~DvProfiler(){}

    //req_start
    //process_start
    //classify_start, classify_end
    //detect_start, detect_end
    //color_start, color_end
    //plate_start, plate_end
    //marker_start, marker_end
    //feature_start, feature_end
    //process_end
    //req_end
    void update_profile(map<string, uint64_t>& m, int images, bool success)
    {
        requests_++;
        images_ += images;
        if (!success) {
            fails_++;
            fimgs_ += images;
            return;
        }

        int valid_images = success ? images : 0;
        total_.update(m, "req_end", "req_start", valid_images);
        prev_proc_.update(m, "process_start", "req_start", valid_images);
        classify_.update(m, "classify_end", "classify_start", valid_images);
        detect_.update(m, "detect_end", "detect_start", valid_images);
        color_.update(m, "color_end", "color_start", valid_images);
        plate_.update(m, "plate_end", "plate_start", valid_images);
        marker_.update(m, "marker_end", "marker_start", valid_images);
        feature_.update(m, "feature_end", "feature_start", valid_images);
        next_proc_.update(m, "req_end", "process_end", valid_images);

        if (gettime() - latest_ > 60000)
        {
            LOG(WARNING) << "requests: " << requests_ << ", images: " << images_ << ", valid requests: " << (requests_ - fails_);
            requests_ = images_ = fails_ = fimgs_ = 0;
            total_.dump();
            prev_proc_.dump();
            classify_.dump();
            detect_.dump();
            color_.dump();
            plate_.dump();
            marker_.dump();
            feature_.dump();
            next_proc_.dump();

            latest_ = gettime();
        }
    }

private:
    uint64_t latest_;
    int requests_, images_, fails_, fimgs_;
    DVProfStats total_, prev_proc_, classify_, detect_, color_, plate_, marker_, feature_, next_proc_;

    uint64_t gettime()
    {
        struct timeval n;
        gettimeofday(&n, NULL);
        return n.tv_sec * 1000 + n.tv_usec / 1000; //ms
    }
};
}
#endif //PROJECT_TIME_PROFILER_H
