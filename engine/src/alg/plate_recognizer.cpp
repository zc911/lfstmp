/*
 * plate_recognizer.cpp
 *
 *  Created on: Apr 27, 2016
 *      Author: jiajaichen
 */

#include "plate_recognizer.h"

namespace dg {

const int WIDTH = 2560;
const int HEIGHT = 2048;

PlateRecognizer::PlateRecognizer(const PlateConfig &config) {
    mem1 = new unsigned char[0x4000];
    mem2 = new unsigned char[40 * 1024 * 1024];
    c_Config = c_defConfig;
    c_Config.nMinPlateWidth = config.MinWidth;
    c_Config.nMaxPlateWidth = config.MaxWidth;
    c_Config.bMovingImage = config.IsMovingImage;
    c_Config.bOutputSingleFrame = 1;
    c_Config.nMaxImageWidth = WIDTH;
    c_Config.nMaxImageHeight = HEIGHT;
    c_Config.pFastMemory = mem1;
    c_Config.nFastMemorySize = 0x4000;
    c_Config.pMemory = mem2;
    c_Config.nMemorySize = 40 * 1024 * 1024;
    c_Config.bIsFieldImage = 0;
    c_Config.bUTF8 = 1;

    int nRet = TH_InitPlateIDSDK(&c_Config);
    LOG(INFO)<<"nREt: "<<nRet<<endl;
    if (nRet != 0) {
        LOG(INFO)<<("nRet = %d, try sudo ./program\n", nRet)<<endl;
        exit(-1);
    }
    nRet = TH_SetProvinceOrder((char *) config.LocalProvince.c_str(),
                               &c_Config);
    nRet = TH_SetRecogThreshold(5, 2, &c_Config);
    LOG(INFO)<<"TH_SetRecogThreshold "<<nRet<<endl;
    nRet = TH_SetImageFormat(ImageFormatBGR, 0, 0, &c_Config);
    LOG(INFO)<<"TH_SetImageFormat "<<nRet<<endl;
int threads=1;
stop=false;
    for(size_t i = 0;i<threads;++i)
        workers.emplace_back(
            [this]
            {
                for(;;)
                {
                    std::function<void()> task;

                    {
                        std::unique_lock<std::mutex> lock(this->queue_mutex);
                        this->condition.wait(lock,
                            [this]{ return this->stop || !this->tasks.empty(); });
                        if(this->stop && this->tasks.empty())
                            return;
                        task = std::move(this->tasks.front());
                        this->tasks.pop();
                    }

                    task();
                }
            }
        );
  //  tp_.SetSize(1);
}

PlateRecognizer::~PlateRecognizer() {
    delete mem1;
    delete mem2;
    {
         std::unique_lock<std::mutex> lock(queue_mutex);
         stop = true;
     }
     condition.notify_all();
     for(std::thread &worker: workers)
         worker.join();
}
template<class F, class... Args>
auto PlateRecognizer::enqueue(F&& f, Args&&... args)
    -> std::future<typename std::result_of<F(Args...)>::type>
{
    using return_type = typename std::result_of<F(Args...)>::type;

    auto task = std::make_shared< std::packaged_task<return_type()> >(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );

    std::future<return_type> res = task->get_future();
    {
        std::unique_lock<std::mutex> lock(queue_mutex);

        // don't allow enqueueing after stopping the pool
        if(stop)
            throw std::runtime_error("enqueue on stopped ThreadPool");

        tasks.emplace([task](){ (*task)(); });
    }
    condition.notify_one();
    return res;
}

int PlateRecognizer::recognizeImage(const Mat &img) {
    Mat sample;

    if (img.channels() == 4)
        cvtColor(img, sample, CV_BGRA2BGR);
    else if (img.channels() == 1)
        cvtColor(img, sample, CV_GRAY2BGR);
    else
        sample = img;

    if (sample.channels() != 3) {
        LOG(INFO)<<"Sample color error"<<sample.channels()<<endl;
    }

    // TODO resize if image too large
//    if ((sample.rows > 1000) || (sample.cols > 1000)) {
//        return -1;
//    }

    unsigned char *pImg = new unsigned char[sample.rows * sample.cols * 3];

    int cnt = 0;
    for (int i = 0; i < sample.rows; i++) {
        for (int j = 0; j < sample.cols; j++) {
            for (int c = 0; c < sample.channels(); c++) {
                pImg[cnt++] = sample.at<uchar>(i, j * 3 + c);
            }
        }
    }

    int nResultNum = 1;

    int nRet = TH_RecogImage(pImg, sample.cols, sample.rows, &result,
                             &nResultNum, NULL, &c_Config);

    delete[] pImg;

    if (nRet != 0) {
        LOG(WARNING)<<"Plate recognizer error : "<<nRet<<endl;
    }
    return nRet;
}
int r(const Mat &img,TH_PlateIDResult &result,TH_PlateIDCfg *c_Config) {
    Mat sample;

    if (img.channels() == 4)
        cvtColor(img, sample, CV_BGRA2BGR);
    else if (img.channels() == 1)
        cvtColor(img, sample, CV_GRAY2BGR);
    else
        sample = img;

    if (sample.channels() != 3) {
        LOG(INFO)<<"Sample color error"<<sample.channels()<<endl;
    }

    // TODO resize if image too large
//    if ((sample.rows > 1000) || (sample.cols > 1000)) {
//        return -1;
//    }

    unsigned char *pImg = new unsigned char[sample.rows * sample.cols * 3];

    int cnt = 0;
    for (int i = 0; i < sample.rows; i++) {
        for (int j = 0; j < sample.cols; j++) {
            for (int c = 0; c < sample.channels(); c++) {
                pImg[cnt++] = sample.at<uchar>(i, j * 3 + c);
            }
        }
    }

    int nResultNum = 1;

    int nRet = TH_RecogImage(pImg, sample.cols, sample.rows, &result,
                             &nResultNum, NULL, c_Config);
    delete[] pImg;

    if (nRet != 0) {
        LOG(WARNING)<<"Plate recognizer error : "<<nRet<<endl;
    }
    return nRet;
}
Vehicle::Plate PlateRecognizer::Recognize(const Mat &img) {
    if (nRet != 0) {
        LOG(INFO)<<"plate recognizer error : "<<nRet<<endl;
    }

    recognizeImage(img);

    Vehicle::Plate plate;
    plate.plate_num = result.license;
    plate.color_id = result.nColor;
    plate.plate_type = result.nType;
    plate.confidence = result.nConfidence / 100.0;

    Box cutboard;
    cutboard.x=result.rcLocation.left;
    cutboard.y=result.rcLocation.top;
    cutboard.width=result.rcLocation.right-result.rcLocation.left;
    cutboard.height=result.rcLocation.bottom-result.rcLocation.top;
    plate.box=cutboard;
    return plate;
}

vector<Vehicle::Plate> PlateRecognizer::RecognizeBatch(
        const vector<Mat> &imgs) {

    vector<Vehicle::Plate> vRecognizeResult;

    TH_PlateIDCfg *config=&c_Config;
    int imagesize = imgs.size();
    int batchsize=1;

    for (int i = 0; i < (ceil((float)imagesize / (float)batchsize) * batchsize); i +=
            batchsize){
        vector<future<int> >ress;
        TH_PlateIDResult result;
        vector<TH_PlateIDResult*> results;
        for (int j = 0; j < batchsize; j++) {
            if (i * batchsize + j < imagesize) {
                Mat img = imgs[i * batchsize + j];
                results.push_back(&result);
                ress.emplace_back(this->enqueue([img,&result,config]() {return r(img,result,config);}));
            }
        }
        for(auto && res: ress){

            res.get();

        }
        for(int i=0;i<results.size();i++){
            TH_PlateIDResult *tmp=results[i];
            Vehicle::Plate plate;
            plate.plate_num = tmp->license;
            plate.color_id = tmp->nColor;
            plate.plate_type = tmp->nType;
            plate.confidence = tmp->nConfidence / 100.0;

            Box cutboard;
            cutboard.x=tmp->rcLocation.left;
            cutboard.y=tmp->rcLocation.top;
            cutboard.width=tmp->rcLocation.right-tmp->rcLocation.left;
            cutboard.height=tmp->rcLocation.bottom-tmp->rcLocation.top;
            vRecognizeResult.push_back(plate);
        }

    }


    return vRecognizeResult;
}
void PlateRecognizer::Init(void *config) {
    int nthreads;
    TH_GetKeyMaxThread(&nthreads);

}

} /* namespace dg */
