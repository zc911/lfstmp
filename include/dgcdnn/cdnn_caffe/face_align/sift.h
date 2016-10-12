#pragma once
#include <vector>
#include "shapevec.h"
using namespace std;
using namespace cv;
const float  SIFT_SIGMA = 1.6f;		    /** default sigma for initial gaussian smoothing */
const int    SIFT_DESCR_WIDTH = 4;		/** default width of descriptor histogram array */
const int    SIFT_DESCR_HIST_BINS = 8;	/** default number of bins per histogram in descriptor array */
const float  SIFT_INIT_SIGMA = 0.5f;		/* assumed gaussian blur for input image */
const float  SIFT_DESCR_MAG_THR = 0.2f;
const float  SIFT_DESCR_SCL_FCTR = 3.0f; /* determines the size of a single descriptor orientation histogram */
const int    FEATURE_MAX_D = 128;
struct Feature
{
  float x;                      /**< x coord */
  float y;                      /**< y coord */
  float scl;                    /**< scale of a Lowe-style feature */
  float ori;                    /**< orientation of a Lowe-style feature */
  int   dim;                    /**< descriptor length */
  vector<float> descr;          /**< descriptor [FEATURE_MAX_D]*/
  Feature(){descr.resize(FEATURE_MAX_D);}
};
class SIFT_Desc
{
public:
  void initImg(Mat_<uchar>& img,Mat_<float>& magnitude,Mat_<float>& angle);
  void SIFT_descriptor(Mat_<uchar>& img, vector<Point>& pts,Mat_<float>& f);
  void SIFT_descriptor(Mat_<uchar>& img, ShapeVec& shape,ShapeVec& refshape,Mat_<float>& f);
  void SIFT_descriptor(Mat_<float>& magnitude,Mat_<float>& angle, ShapeVec& shape,ShapeVec& refshape,Mat_<float>& f);
  int getDim(){return FEATURE_MAX_D;};
//private:
  void compute(Mat_<float>& magnitude,Mat_<float>& angle);
  void compute(Mat_<uchar>& img);
  void descr_hist(Mat_<float>& magnitude,Mat_<float>& angle);
  void descr_hist(Mat_<float>& magnitude,Mat_<float>& angle,Feature& keypoint);
  void normalize_descr(Feature& feat );
  //root sift will increase the performance of sift
  void rootSIFT(Mat_<float>& f)
  {
    //rootsift= sqrt( sift / sum(sift) );
    double norm_dist = norm(f,NORM_L1);
    if(abs(norm_dist)<=FLT_EPSILON)
      norm_dist =1;
    f /= norm_dist;
    sqrt(f,f);
  }
  float scl;
  float ori;
  vector<Feature> keypoints;
};


