#pragma once

#include <iostream>
#include <string>
#include <cv.h>
#include <highgui.h>
#include "shapevec.h"
using namespace std;
using namespace cv;
namespace Cdnn{
#define SQR(a) ((a)*(a)) 
//const int template_width  = 128;  //this is the size of the warped image, task independent. just set as constant value here
//const int template_height = 128;
//const int CENTER_IN_TEMPLATE_X = template_width*0.5;
//const int CENTER_IN_TEMPLATE_Y = template_height*0.5;


float calculateOri(const ShapeVec& shape, const  ShapeVec& refshape);

//for component extraction
void getComponent(vector< Point_<double> >& shape,int comp_idx,vector< Point_<double> >& component_pts);
void setComponent(vector< Point_<double> >& shape,int comp_idx,vector< Point_<double> >& component_pts);
void getMeanShape(vector< Point_<double> >& meanShape,int nPoints);
void getComponentImg(Mat& img,vector< Point_<double> >& shape,int comp_idx,Mat& comp_img,vector< Point_<double> >& comp_pts,vector< Mat_<double> >& dstMaps, int template_width);
//
void getMean(vector< Point_<double> >& shape, Point_<double>& Gravity);
void doTranslate(vector< Point_<double> >& shape, Point_<double>& t);
void doScale(vector< Point_<double> >& shape,double r);
void zeroGravity(vector< Point_<double> >& shape);
void scaleToOne(vector< Point_<double> >& shape);
void boundingBox(vector< Point_<double> >& shape,Rect_<double>& box);
//
void getPointSetCenterScale(Mat_<float>& shape,Point2f& center, float& scale);
void getPointSetCenterScaleOri(Mat_<float>& shape,int pt_id1,int pt_id2, Point2f& center, float& scale,float& tilt_angle);
void HFlipsample(Mat& srcImg,vector< Point_<double> >& srcPt,Mat& dstImg,vector< Point_<double> >& dstPt);
inline float InvSqrt(float x)
{
    float xhalf = 0.5f*x;
    int i = *(int*)&x;       // get bits for floating value
    i = 0x5f375a86- (i>>1);  // gives initial guess y0
    x = *(float*)&i;         // convert bits back to float
    x = x*(1.5f-xhalf*x*x);  // Newton step, repeating increases accuracy
    return x;
}
inline float fastexp7(float x)
{
    return (5040+x*(5040+x*(2520+x*(840+x*(210+x*(42+x*(7+x)))))))*0.00019841269f;
}
//Affine mapping functions
inline void Affine_Point(Mat_<double> &M,const Point_<int>& srcPt, Point_<int> &dstPt)
{
    int x = int(M(0,0)*srcPt.x + M(0,1)*srcPt.y + M(0,2) +0.5);
    int y = int(M(1,0)*srcPt.x + M(1,1)*srcPt.y + M(1,2)+ 0.5);
    dstPt.x = x;
    dstPt.y = y;
}
inline void Affine_Point(Mat_<double> &M,const Point_<double>& srcPt, Point_<double> &dstPt)
{
    int x = int(M(0,0)*srcPt.x + M(0,1)*srcPt.y + M(0,2) +0.5);
    int y = int(M(1,0)*srcPt.x + M(1,1)*srcPt.y + M(1,2)+ 0.5);
    dstPt.x = x;
    dstPt.y = y;
}
inline Mat_<double> Get_Affine_matrix(Point_<double>& srcCenter, Point_<double>& dstCenter,double alpha, double scale)
{
    Mat_<double> M(2,3);

    M(0,0) = scale*cos(alpha);
    M(0,1) = scale*sin(alpha);
    M(1,0) = -M(0,1);
    M(1,1) =  M(0,0);

    M(0,2) = srcCenter.x - M(0,0)*dstCenter.x - M(0,1)*dstCenter.y;
    M(1,2) = srcCenter.y - M(1,0)*dstCenter.x - M(1,1)*dstCenter.y;
    return M;
}
inline Mat_<double> inverseMatrix(Mat_<double>& M)
{
     double D = M(0,0)*M(1,1) - M(0,1)*M(1,0);
    D = D != 0 ? 1./D : 0;

    Mat_<double> inv_M(2,3);

    inv_M(0,0) = M(1,1)*D;
    inv_M(0,1) = M(0,1)*(-D);
    inv_M(1,0) = M(1,0)*(-D);
    inv_M(1,1) = M(0,0)*D;

    inv_M(0,2) = -inv_M(0,0)*M(0,2) - inv_M(0,1)*M(1,2);
    inv_M(1,2) = -inv_M(1,0)*M(0,2) - inv_M(1,1)*M(1,2);
    return inv_M;
}
void mAffineWarp(const Mat_<float> M, const Mat& srcImg, Mat& dstImg);
void getWarpingMap(vector< Point_<double> >& shape,vector< Point_<double> >& refshape,Mat_<double>& M);
void AffineWarpImage(const Mat_<float> M, const IplImage* srcImg, IplImage* dstImg);

}