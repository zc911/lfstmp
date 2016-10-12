#pragma once
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <vector>
using namespace std;
using namespace cv;
//2013-3-16
//! Shape Vector
namespace Cdnn{
class ShapeVec : public Mat_< float >
{
public:
    ShapeVec(){}
    ShapeVec(const Mat_< float > &a):Mat_< float >(a){}
    ShapeVec & operator =(const Mat_< float > &a) 
    {
        Mat_< float >::operator=(a);
        return *this;
    }
     //! Align to another shape vector
    void alignTo(const ShapeVec & ref);
    
    //! Move the center of gravity to origin. X and Y are moved seperately.
    void zeroGravity();
    
    //! Scale the vector's norm to 1.
    void scaleToOne();
    
    void doTranslate(double vX, double vY);
    void doScale(double r);
    
    double getXMean() const{return mean(rowRange(0, rows/2))[0];}
    double getYMean() const{return mean(rowRange(rows/2, rows))[0];}

    float X(int i) const{ return (*this)(i, 0); }
    float & X(int i) { return (*this)(i, 0); }
    float Y(int i) const{ return (*this)(i+(rows>>1), 0); }
    float & Y(int i) { return (*this)(i+(rows>>1), 0); }
    void fromPointList(const vector<Point > &v);
    void toPointList(vector< Point > &v);
    void toPointList(vector< Point_<double> > &v);
    int nPoints() const { return (rows>>1); }
};

//! A similarity transformation
class SimilarityTrans
{
public:
    void backTransform(const ShapeVec &src, ShapeVec &dst);
    void transform(const ShapeVec &src, ShapeVec &dst) const;
    void setTransformByAlign(const ShapeVec &x, const ShapeVec &xp);
    void warpImage(const Mat &imgSrc, Mat &imgDst);
    void warpImgBack(const Mat &imgSrc, Mat &imgDst, bool useDstSize=false);
    //! Get the scale factor
    double getS() const { return sqrt(a*a+b*b); }
    
    SimilarityTrans():Xt(0), Yt(0), a(1), b(0){}
    
    SimilarityTrans operator *(const SimilarityTrans & s2)
    {
        SimilarityTrans ns;
        ns.a = a*s2.a-b*s2.b;
        ns.b = s2.a*b+s2.b*a;
        ns.Xt = a*s2.Xt - b*s2.Yt + Xt;
        ns.Yt = b*s2.Xt + a*s2.Yt + Yt;
        return ns;
    }
// private:
    double Xt;//! X translate
    double Yt; //! Y translate
    double a;//! a in similarity transformation matrix
    double b;//! b in similarity transformation matrix
}; 

}