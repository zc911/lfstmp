#pragma once
#include <iostream>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include "shapevec.h"
using namespace std;
using namespace cv;

#define SQR(a) ((a)*(a)) 
#include <sys/time.h>//for  gettimeofday()
using namespace std;
//timer for benchmark time useage
class timer
{
    public:
        double start, startu;
        void tic()
        {
            struct timeval tp;
            gettimeofday(&tp, NULL);
            start = tp.tv_sec;
            startu = tp.tv_usec;
        }
        double toc()
        {
            struct timeval tp;
            gettimeofday(&tp, NULL);
            return( ((double) (tp.tv_sec - start)) + (tp.tv_usec-startu)/1000000.0 );
        }
};

vector<string> readlines(string& filename);
float calculateOri(const ShapeVec& shape, const  ShapeVec& refshape);

//for component extraction
void getComponent(vector< Point_<float> >& shape,int comp_idx,vector< Point_<float> >& component_pts);
void setComponent(vector< Point_<float> >& shape,int comp_idx,vector< Point_<float> >& component_pts);
void getMeanShape(vector< Point_<float> >& meanShape,int nPoints);
void getComponentImg(Mat& img,vector< Point_<float> >& shape,int comp_idx,Mat& comp_img,vector< Point_<float> >& comp_pts,vector< Mat_<float> >& dstMaps, int template_width);

void getPointSetCenterScale(Mat_<float>& shape,Point2f& center, float& scale);
void getPointSetCenterScaleOri(Mat_<float>& shape,int pt_id1,int pt_id2, Point2f& center, float& scale,float& tilt_angle);
void HFlipsample(Mat& srcImg,vector< Point_<float> >& srcPt,Mat& dstImg,vector< Point_<float> >& dstPt);
 void getRotRect(vector<Point2f>& landmarks,RotatedRect& box);
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
template<typename T>  
inline void Affine_Point(const Mat_<float> &M,T& srcPt, T &dstPt)
{
    float x = M(0,0)*srcPt.x + M(0,1)*srcPt.y + M(0,2);
    float y = M(1,0)*srcPt.x + M(1,1)*srcPt.y + M(1,2);
    dstPt.x = x;
    dstPt.y = y;
}
inline void Affine_Shape(const Mat_<float>& M,const ShapeVec& srcShape, ShapeVec& dstShape)
{
    int N = srcShape.nPoints();
    if(dstShape.rows!=srcShape.rows||dstShape.cols!=srcShape.cols)
        dstShape.create(srcShape.rows,srcShape.cols);
    for(int i=0;i<N;++i)
    {
        Point_<float> pt(srcShape.X(i),srcShape.Y(i));
        Affine_Point(M,pt,pt);
        dstShape.X(i) = (float)pt.x;
        dstShape.Y(i) = (float)pt.y;
    }
}

template<typename T> 
inline Mat_<float> Get_Affine_matrix(T& srcCenter, T& dstCenter,float alpha, float scale)
{
    Mat_<float> M(2,3);

    M(0,0) = scale*cos(alpha);
    M(0,1) = scale*sin(alpha);
    M(1,0) = -M(0,1);
    M(1,1) =  M(0,0);

    M(0,2) = srcCenter.x - M(0,0)*dstCenter.x - M(0,1)*dstCenter.y;
    M(1,2) = srcCenter.y - M(1,0)*dstCenter.x - M(1,1)*dstCenter.y;
    return M;
}
inline Mat_<float> inverseMatrix(Mat_<float>& M)
{
     float D = M(0,0)*M(1,1) - M(0,1)*M(1,0);
    D = D != 0 ? 1./D : 0;

    Mat_<float> inv_M(2,3);

    inv_M(0,0) = M(1,1)*D;
    inv_M(0,1) = M(0,1)*(-D);
    inv_M(1,0) = M(1,0)*(-D);
    inv_M(1,1) = M(0,0)*D;

    inv_M(0,2) = -inv_M(0,0)*M(0,2) - inv_M(0,1)*M(1,2);
    inv_M(1,2) = -inv_M(1,0)*M(0,2) - inv_M(1,1)*M(1,2);
    return inv_M;
}
void mAffineWarp(const Mat_< float >& M, 
        const Mat& srcImg,
        Mat& dstImg, 
        int interpolation = INTER_LINEAR);
void getWarpingMap(vector< Point_<float> >& shape,vector< Point_<float> >& refshape,Mat_<float>& M);
Mat_<float> getWarpingMap(ShapeVec& shape,ShapeVec& refshape);
//point set process functions
template<typename T> 
void getMean(vector< T >& shape, T& Gravity)
{
    Gravity.x =0;
    Gravity.y =0;
    for(int i=0;i<shape.size();++i)
        Gravity += shape[i];
    Gravity.x /=  shape.size();
    Gravity.y /=  shape.size();

} 
template<typename T> 
void doTranslate(vector< T >& shape, T& t)
{
    for(int i=0;i<shape.size();++i)
        shape[i]+=t;
}
template<typename T> 
void doScale(vector< T >& shape,float r)
{
    for(int i=0;i<shape.size();++i)
    {
        shape[i].x *= r;
        shape[i].y *= r;
    }
}
template<typename T> 
void zeroGravity(vector< T >& shape)
{
    T Gravity;
    getMean(shape,Gravity);
    for(int i=0;i<shape.size();++i)
        shape[i] -= Gravity;
}
template<typename T> 
void scaleToOne(vector< T >& shape)
{
    float norm_value = 0;
    for(int i=0;i<shape.size();++i)
        norm_value += SQR(shape[i].x)+SQR(shape[i].y);
    norm_value = 1.0/sqrt(norm_value);
    doScale(shape,norm_value);
}
template<typename T> 
void boundingBox(vector< T >& shape,Rect& box)
{
    float min_x = 10e10;
    float min_y = 10e10;
    float max_x = -10e10;
    float max_y = -10e10;
    for(int i=0;i<shape.size();++i)
    {
        min_x = min_x > shape[i].x ? shape[i].x : min_x;
        min_y = min_y > shape[i].y ? shape[i].y : min_y;
        max_x = max_x < shape[i].x ? shape[i].x : max_x;
        max_y = max_y < shape[i].y ? shape[i].y : max_y;
    }
    box.x = int(min_x);
    box.y = int(min_y);
    box.width =  int(max_x - min_x);
    box.height =  int(max_y - min_y);
}
template<typename T>
void getBoundingBox(vector<T>& shape, T& TL, T& BR)
{
    float min_x = 10e10;
    float min_y = 10e10;
    float max_x = 0;
    float max_y = 0;
    for (int i = 0; i < shape.size(); ++i)
    {
        if (shape[i].x < min_x)
            min_x = shape[i].x;
        if (shape[i].y < min_y)
            min_y = shape[i].y;
        if (shape[i].x > max_x)
            max_x = shape[i].x;
        if (shape[i].y > max_y)
            max_y = shape[i].y;
    }
    TL.x = min_x - 1;
    TL.y = min_y - 1;
    BR.x = max_x + 1;
    BR.y = max_y + 1;
}
