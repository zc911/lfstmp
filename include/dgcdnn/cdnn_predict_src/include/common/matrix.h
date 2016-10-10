
#ifndef MATRIX_H_
#define MATRIX_H_

#ifdef NUMPY_INTERFACE
#include <Python.h>
#include <arrayobject.h>
#endif
#include <limits>
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "matrix_funcs.h"
#include "matrix_operators.h"

#ifdef USE_ATLAS
//for atlas
extern "C" {
#include <cblas.h>
}
#endif

#ifdef USE_MKL
//for mkl
#include <mkl_cblas.h>
#endif

#ifdef DOUBLE_PRECISION
#define CBLAS_GEMM cblas_dgemm
#define CBLAS_SCAL cblas_dscal
#define CBLAS_AXPY cblas_daxpy
#else
#define CBLAS_GEMM cblas_sgemm
#define CBLAS_SCAL cblas_sscal
#define CBLAS_AXPY cblas_saxpy
#endif /* DOUBLE_PRECISION */

#define MTYPE_MAX numeric_limits<MTYPE>::max()

class Matrix {
private:
    MTYPE* _data;
    bool _ownsData;
    long int _numRows, _numCols;
    long int _numElements;
    long int _stride;
    int _imgHeight, _imgWidth;
    CBLAS_TRANSPOSE _trans;

    void _init(MTYPE* data, long int numRows, long int numCols, bool transpose, bool ownsData);
    void _tileTo2(Matrix& target) const;
    void _copyAllTo(Matrix& target) const;
    MTYPE _sum_column(long int col) const;
    MTYPE _sum_row(long int row) const;
    MTYPE _aggregate(MTYPE(*agg_func)(MTYPE, MTYPE), MTYPE initialValue) const;
    void _aggregate(long int axis, Matrix& target, MTYPE(*agg_func)(MTYPE, MTYPE), MTYPE initialValue) const;
    MTYPE _aggregateRow(long int row, MTYPE(*agg_func)(MTYPE, MTYPE), MTYPE initialValue) const;
    MTYPE _aggregateCol(long int row, MTYPE(*agg_func)(MTYPE, MTYPE), MTYPE initialValue) const;
    void _updateDims(long int numRows, long int numCols);
    template <class Op> void _applyLoop(Op op, Matrix& target) {
        MTYPE *ptr = this->_data, *tgtPtr = target._data;
        for (long int i = 0; i < getNumElements(); i++, ptr++, tgtPtr++) {
            *tgtPtr = op(*ptr);
        }
    }
    void _applyLoop(MTYPE(*func)(MTYPE));
    void _applyLoop(MTYPE (*func)(MTYPE), Matrix& target);
    void _applyLoop2(const Matrix& a, MTYPE(*func)(MTYPE, MTYPE), Matrix& target) const;
    void _applyLoop2(const Matrix& a, MTYPE (*func)(MTYPE,MTYPE, MTYPE), MTYPE scalar, Matrix& target) const;
    void _applyLoopScalar(const MTYPE scalar, MTYPE(*func)(MTYPE, MTYPE), Matrix& target) const;
    void _checkBounds(long int startRow, long int endRow, long int startCol, long int endCol) const;
    void _divideByVector(const Matrix& vec, Matrix& target);
    inline long int _getNumColsBackEnd() const {
        return _trans == CblasNoTrans ? _numCols : _numRows;
    }
public:
    enum FUNCTION {
        TANH, RECIPROCAL, SQUARE, ABS, EXP, LOG, ZERO, ONE, LOGISTIC1, LOGISTIC2, SIGN
    };
    Matrix();
    Matrix(long int numRows, long int numCols);
#ifdef NUMPY_INTERFACE
    Matrix(const PyArrayObject *src);
#endif
    Matrix(const Matrix &like);
    Matrix(MTYPE* data, long int numRows, long int numCols);
    Matrix(MTYPE* data, long int numRows, long int numCols, bool transpose, bool ownsData);
    ~Matrix();

    inline MTYPE& getCell(long int i, long int j) const {
        assert(i >= 0 && i < _numRows);
        assert(j >= 0 && j < _numCols);
        if (_trans == CblasTrans) {
            return _data[j * _numRows + i];
        }
        return _data[i * _numCols + j];
    }

    MTYPE& operator()(long int i, long int j) const {
        return getCell(i, j);
    }

    inline MTYPE* getData() const {
        return _data;
    }

    inline bool isView() const {
        return !_ownsData;
    }

    inline long int getNumRows() const {
        return _numRows;
    }

    inline long int getNumCols() const {
        return _numCols;
    }
    
    inline long int getNumDataBytes() const {
        return _numElements * sizeof(MTYPE);
    }

    inline long int getNumElements() const {
        return _numElements;
    }

    inline long int getLeadingDim() const {
        return _trans == CblasTrans ? _numRows : _numCols;
    }

    inline long int getFollowingDim() const {
        return _trans == CblasTrans ? _numCols : _numRows;
    }

    inline CBLAS_TRANSPOSE getBLASTrans() const {
        return _trans;
    }

    inline bool isSameDims(const Matrix& a) const {
        return a.getNumRows() == getNumRows() && a.getNumCols() == getNumCols();
    }

    inline bool isTrans() const {
        return _trans == CblasTrans;
    }

    /*
     * Only use if you know what you're doing!
     * Does not update any dimensions. Just flips the _trans flag.
     *
     * Use transpose() if you want to get the transpose of this matrix.
     */
    inline void setTrans(bool trans) {
        assert(isTrans() == trans || !isView());
        _trans = trans ? CblasTrans : CblasNoTrans;
    }
    
    inline void zeroMem() {
        if (_data) {
            memset(_data, 0, _numElements * sizeof(float));
        }
    }
    void setImgHeight(int height) {
        _imgHeight = height;
    }
    void setImgWidth(int width) {
        _imgWidth = width;
    }
    int getImgHeight() {
        return _imgHeight;
    }
    int getImgWidth() {
      return _imgWidth;
    }
    void setData(MTYPE *data) {
        _data = data;
    }
    void apply(FUNCTION f);
    void apply(Matrix::FUNCTION f, Matrix& target);
    template <class Op> void apply(Op op) {
        apply(op, *this);
    }
    template <class Op> void apply(Op op, Matrix& target){
        this->_applyLoop(op, target);
    }
    void subtractFromScalar(MTYPE scalar);
    void subtractFromScalar(MTYPE scalar, Matrix &target) const;
    void biggerThanScalar(MTYPE scalar);
    void smallerThanScalar(MTYPE scalar);
    void equalsScalar(MTYPE scalar);
    void biggerThanScalar(MTYPE scalar, Matrix& target) const;
    void smallerThanScalar(MTYPE scalar, Matrix& target) const;
    void equalsScalar(MTYPE scalar, Matrix& target) const;
    void biggerThan(Matrix& a);
    void biggerThan(Matrix& a, Matrix& target) const;
    void smallerThan(Matrix& a);
    void smallerThan(Matrix& a, Matrix& target) const;
    void minWith(Matrix &a);
    void minWith(Matrix &a, Matrix &target) const;
    void maxWith(Matrix &a);
    void maxWith(Matrix &a, Matrix &target) const;
    void equals(Matrix& a);
    void equals(Matrix& a, Matrix& target) const;
    void notEquals(Matrix& a) ;
    void notEquals(Matrix& a, Matrix& target) const;
    void add(const Matrix &m);
    void add(const Matrix &m, MTYPE scale);
    void add(const Matrix &m, Matrix& target);
    void add(const Matrix &m, MTYPE scale, Matrix& target);
    void subtract(const Matrix &m);
    void subtract(const Matrix &m, Matrix& target);
    void subtract(const Matrix &m, MTYPE scale);
    void subtract(const Matrix &m, MTYPE scale, Matrix& target);
    void addVector(const Matrix& vec, MTYPE scale);
    void addVector(const Matrix& vec, MTYPE scale, Matrix& target);
    void addVector(const Matrix& vec);
    void addVector(const Matrix& vec, Matrix& target);
    void addScalar(MTYPE scalar);
    void addScalar(MTYPE scalar, Matrix& target) const;
    void maxWithScalar(MTYPE scalar);
    void maxWithScalar(MTYPE scalar, Matrix &target) const;
    void minWithScalar(MTYPE scalar);
    void minWithScalar(MTYPE scalar, Matrix &target) const;
    void eltWiseMultByVector(const Matrix& vec);
    void eltWiseMultByVector(const Matrix& vec, Matrix& target);
    void eltWiseDivideByVector(const Matrix& vec);
    void eltWiseDivideByVector(const Matrix& vec, Matrix& target);
    void resize(long int newNumRows, long int newNumCols);
    void resize(const Matrix& like);
    Matrix& slice(long int startRow, long int endRow, long int startCol, long int endCol) const;
    void slice(long int startRow, long int endRow, long int startCol, long int endCol, Matrix &target) const;
    Matrix& sliceRows(long int startRow, long int endRow) const;
    void sliceRows(long int startRow, long int endRow, Matrix& target) const;
    Matrix& sliceCols(long int startCol, long int endCol) const;
    void sliceCols(long int startCol, long int endCol, Matrix& target) const;
    void rightMult(const Matrix &b, MTYPE scale);
    void rightMult(const Matrix &b, Matrix &target) const;
    void rightMult(const Matrix &b);
    void rightMult(const Matrix &b, MTYPE scaleAB, Matrix &target) const;
    void addProduct(const Matrix &a, const Matrix &b, MTYPE scaleAB, MTYPE scaleThis);
    void addProduct(const Matrix& a, const Matrix& b);
    void eltWiseMult(const Matrix& a);
    void eltWiseMult(const Matrix& a, Matrix& target) const;
    void eltWiseDivide(const Matrix& a);
    void eltWiseDivide(const Matrix& a, Matrix &target) const;
    Matrix& transpose() const;
    Matrix& transpose(bool hard) const;
    void transpose();
    void reverseBlasTrans();
    Matrix& tile(long int timesY, long int timesX) const;
    void tile(long int timesY, long int timesX, Matrix& target) const;
    void copy(Matrix &dest, long int srcStartRow, long int srcEndRow, long int srcStartCol, long int srcEndCol, long int destStartRow, long int destStartCol) const;
    Matrix& copy() const;
    void copy(Matrix& target) const;
    Matrix& sum(long int axis) const;
    void sum(long int axis, Matrix &target) const;
    MTYPE sum() const;
    MTYPE max() const;
    Matrix& max(long int axis) const;
    void max(long int axis, Matrix& target) const;
    MTYPE min() const;
    Matrix& min(long int axis) const;
    void min(long int axis, Matrix& target) const;
    MTYPE norm() const;
    MTYPE norm2() const;
    void scale(MTYPE scale);
    void scale(MTYPE alpha, Matrix& target);
    void reshape(long int numRows, long int numCols);
    Matrix& reshaped(long int numRows, long int numCols);
    void printShape(const char* name) const;
    bool hasNan() const;
    bool hasInf() const;
    void randomizeNormal(MTYPE mean, MTYPE stdev);
    void randomizeUniform();
    void randomizeNormal();
    void print() const;
    void print(long int startRow,long int rows, long int startCol,long int cols) const;
    void print(long int rows, long int cols) const;

    void eltwiseDivideByVector(Matrix &v);
            
};

#endif /* MATRIX_H_ */
