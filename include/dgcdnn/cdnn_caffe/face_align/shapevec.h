#pragma once
#include "opencv2/opencv.hpp"
#include <vector>
using namespace std;
using namespace cv;
//! Shape Vector
class ShapeVec : public Mat_<float>
{
public:
    ShapeVec(){}
    ShapeVec(const Mat_< float > &a) :Mat_< float >(a){}
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

    double getXMean() const{ return mean(rowRange(0, rows / 2))[0]; }
    double getYMean() const{ return mean(rowRange(rows / 2, rows))[0]; }
    template<typename T> void getBoundingBox(T& TL, T& BR)
    {
        float min_x = 10e10;
        float min_y = 10e10;
        float max_x = 0;
        float max_y = 0;
        int N = rows >> 1;
        for (int i = 0; i < N; ++i)
        {
            if ((*this)(i, 0) < min_x)
                min_x = (*this)(i, 0);
            if ((*this)(i + N, 0) < min_y)
                min_y = (*this)(i + N, 0);
            if ((*this)(i, 0) > max_x)
                max_x = (*this)(i, 0);
            if ((*this)(i + N, 0) > max_y)
                max_y = (*this)(i + N, 0);
        }
        TL.x = min_x - 1;
        TL.y = min_y - 1;
        BR.x = max_x + 1;
        BR.y = max_y + 1;
    }
    float X(int i) const{ return (*this)(i, 0); }
    float & X(int i) { return (*this)(i, 0); }
    float Y(int i) const{ return (*this)(i + (rows >> 1), 0); }
    float & Y(int i) { return (*this)(i + (rows >> 1), 0); }

    template<typename T> void fromPointList(const vector< T >& v)
    {
        this->create(v.size() * 2, 1);
        int N = (rows >> 1);
        for (int i = 0; i < N; ++i)
        {
            (*this)(i, 0) = v[i].x;
            (*this)(i + N, 0) = v[i].y;
        }
    }
    template<typename T> void toPointList(vector< T > &v)
    {
        int N = this->nPoints();
        v.resize(N);
        for (int i = 0; i < N; ++i)
        {
            v[i].x = (*this)(i, 0);
            v[i].y = (*this)(i + N, 0);
        }
    }
    
    int nPoints() const { return (rows >> 1); }
};

//! A similarity transformation
class SimilarityTrans
{
public:
    SimilarityTrans() :Xt(0), Yt(0), a(1), b(0)
    {
        M = Mat_<float>(2, 3);
        invM = Mat_<float>(2, 3);
    }
    void backTransform(const ShapeVec &src, ShapeVec &dst);
    void transform(const ShapeVec &src, ShapeVec &dst) const;
    void setTransformByAlign(const ShapeVec &x, const ShapeVec &xp);
    void warpImage(const Mat &imgSrc, Mat &imgDst);
    void warpImgBack(const Mat &imgSrc, Mat &imgDst, bool useDstSize = false);
    //! Get the scale factor
    double getS() const { return sqrt(a*a + b*b); }
    SimilarityTrans operator *(const SimilarityTrans & s2)
    {
        SimilarityTrans ns;
        ns.a = a*s2.a - b*s2.b;
        ns.b = s2.a*b + s2.b*a;
        ns.Xt = a*s2.Xt - b*s2.Yt + Xt;
        ns.Yt = b*s2.Xt + a*s2.Yt + Yt;
        return ns;
    }
  
    //the transform from DST to src
    //alpha = - src_angle / 180.f * CV_PI 
    //scale =  src_scale / dst_scale;
    template<typename T>
    void setTransMatrix(T& srcCenter, T& dstCenter, float alpha, float scale)
    {
        M(0, 0) = scale*cos(alpha);
        M(0, 1) = scale*sin(alpha);
        M(1, 0) = -M(0, 1);
        M(1, 1) = M(0, 0);
        M(0, 2) = srcCenter.x - M(0, 0)*dstCenter.x - M(0, 1)*dstCenter.y;
        M(1, 2) = srcCenter.y - M(1, 0)*dstCenter.x - M(1, 1)*dstCenter.y;
        setInvM();
    }
    void setTransMatrix(ShapeVec& shape, ShapeVec& refshape);
    float calculateOri(ShapeVec& shape, ShapeVec& refshape);
    
    template<typename T>
    void transPoint(T& srcPt, T &dstPt, int  back = false)
    {
        Mat_<float>& transM = back ? invM : M;
        float x = transM(0, 0)*srcPt.x + transM(0, 1)*srcPt.y + transM(0, 2);
        float y = transM(1, 0)*srcPt.x + transM(1, 1)*srcPt.y + transM(1, 2);
        dstPt.x = x;
        dstPt.y = y;
    }
    void transShape(const ShapeVec& srcShape, ShapeVec& dstShape, int  back = false)
    {
        int N = srcShape.nPoints();
        if (dstShape.rows != srcShape.rows || dstShape.cols != srcShape.cols)
            dstShape.create(srcShape.rows, srcShape.cols);
        for (int i = 0; i < N; ++i)
        {
            Point_<float> pt(srcShape.X(i), srcShape.Y(i));
            transPoint(pt, pt, back);
            dstShape.X(i) = (float)pt.x;
            dstShape.Y(i) = (float)pt.y;
        }
    }
    void transImage(const Mat& srcImg, Mat& dstImg, int interpolation = INTER_LINEAR,int back =false);
    Mat_<float> getM() 
    {
        return M;
    };
    Mat_<float> getInvM()
    {
        return invM;
    };

private:
    void setInvM()
    {
        float D = M(0, 0)*M(1, 1) - M(0, 1)*M(1, 0);
        D = D != 0 ? 1. / D : 0;

        invM(0, 0) = M(1, 1)*D;
        invM(0, 1) = M(0, 1)*(-D);
        invM(1, 0) = M(1, 0)*(-D);
        invM(1, 1) = M(0, 0)*D;

        invM(0, 2) = -invM(0, 0)*M(0, 2) - invM(0, 1)*M(1, 2);
        invM(1, 2) = -invM(1, 0)*M(0, 2) - invM(1, 1)*M(1, 2);
    }

private:
    Mat_<float> M;
    Mat_<float> invM;

    double Xt;//! X translate
    double Yt; //! Y translate
    double a;//! a in similarity transformation matrix
    double b;//! b in similarity transformation matrix
};
