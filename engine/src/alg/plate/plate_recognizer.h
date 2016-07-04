/*
 * plate_recognizer.h
 *
 *  Created on: Apr 27, 2016
 *      Author: jiajaichen
 */

#ifndef SRC_ALG_PLATE_RECOGNIZER_H_
#define SRC_ALG_PLATE_RECOGNIZER_H_

#include <thplateid/TH_PlateID.h>
#include <glog/logging.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "model/model.h"
#include "util/thread_pool.h"

using namespace std;
using namespace cv;
namespace dg {

class PlateRecognizer {
public:
    typedef struct {
        string LocalProvince = "";
        int IsMovingImage = 0;
        int MinWidth = 40;
        int MaxWidth = 400;
        int PlateLocate = 5;
        int OCR = 1;
        bool isSharpen;
    } PlateConfig;

    static PlateRecognizer* Instance(const PlateConfig &config);

    virtual ~PlateRecognizer();
    template<class F, class... Args>
    auto enqueue(F&& f, Args&&... args)
        -> std::future<typename std::result_of<F(Args...)>::type>;
    virtual void Init(void *config);

    virtual vector<Vehicle::Plate> RecognizeBatch(const vector<Mat> &imgs);

    virtual Vehicle::Plate Recognize(const Mat &img);

    TH_PlateIDCfg c_Config;
    unsigned char *mem1;
    unsigned char *mem2;
protected:
    TH_PlateIDResult result;
    int nRet = 0;
private:
    PlateRecognizer(const PlateConfig &config);
    int recognizeImage(const Mat &img);
    // need to keep track of threads so we can join them
    std::vector< std::thread > workers;
    // the task queue
    std::queue< std::function<void()> > tasks;

    // synchronization
    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop;
};

} /* namespace dg */

#endif /* SRC_ALG_PLATE_RECOGNIZER_H_ */
