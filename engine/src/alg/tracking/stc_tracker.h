#ifndef SRC_ALG_STC_TRACKER_H_
#define SRC_ALG_STC_TRACKER_H_

#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

namespace dg {

class STCTracker {
public:
    STCTracker();
    ~STCTracker();
    void init(const Mat frame, const Rect inbox, const float resize_rto,
              const float *line);
    void tracking(const Mat frame, Rect &in_out_trackBox);

private:
    void createHammingWin();
    void complexOperation(const Mat src1, const Mat src2, Mat &dst,
                          int flag = 0);
    void getCxtPriorPosteriorModel(const Mat image);
    void learnSTCModel(const Mat image);
    float linefit(const int &y);

private:
    float line_k;
    float line_b;
    float resize_rto_;
    int FrameNum;
    double sigma;            // scale parameter (variance)
    double alpha;            // scale parameter
    double beta;            // shape parameter
    double rho;                // learning parameter
    double scale;            //	scale ratio
    double lambda;        //	scale learning parameter
    int num;                    //	the number of frames for updating the scale
    int box_w, box_h, box_init_y;
    vector<double> maxValue;
    Point center;            //	the object position
    Rect cxtRegion;        // context region
    int padding;

    Mat cxtPriorPro;        // prior probability
    Mat cxtPosteriorPro;    // posterior probability
    Mat STModel;            // conditional probability
    Mat STCModel;            // spatio-temporal context model
    Mat hammingWin;            // Hamming window
};

typedef struct {
    long long id;
    int pyr_level;
    float wh_rto;
    STCTracker tracker;
} STC;

}
#endif /* SRC_ALG_STC_TRACKER_H_ */
