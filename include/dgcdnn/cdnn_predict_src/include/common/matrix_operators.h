
#ifndef NVMATRIX_OPERATORS_H_
#define NVMATRIX_OPERATORS_H_

#include "matrix_funcs.h"

#ifndef DIVUP
#define DIVUP(x, y) (((x) + (y) - 1) / (y))
#endif

class MatrixOps {
public:
    class Exp {
    public:
        inline float operator()(const float a) const {
            return _exp(a);
        }
    };

    class Logistic {
    public:
        inline float operator()(const float a) const {
            return 1.0f / (1.0f + _exp(-a));
        }
    };

    class Log {
    public:
        inline float operator()(const float a) const {
            return _log(a);
        }
    };

    class Square {
    public:
        inline float operator()(const float a) const {
            return a * a;
        }
    };

    class Sqrt {
    public:
        inline float operator()(const float a) const {
            return sqrt(a);
        }
    };

    class Reciprocal {
    public:
        inline float operator()(const float a) const {
            return 1.0f / a;
        }
    };

    class Abs {
    public:
        inline float operator()(const float a) const {
            return a > 0 ? a : -a;
        }
    };

    class Sign {
    public:
        inline float operator()(const float a) const {
            return (a > 0) - (a < 0);
        }
    };
    
    class Identity {
    public:
        inline float operator()(const float a) const {
            return a;
        }
    };

    class Zero {
    public:
        inline float operator()(const float a) const {
            return 0;
        }
    };

    class One {
    public:
        inline float operator()(const float a) const {
            return 1;
        }
    };
    
    class SmallerThanScalar {
    private:
        const float scalar;
    public:
        SmallerThanScalar(const float _scalar) : scalar(_scalar) {
        }
        inline float operator()(const float a) const {
            return a < scalar;
        }
    };

    class BiggerThanScalar {
    private:
        const float scalar;
    public:
        BiggerThanScalar(const float _scalar) : scalar(_scalar) {
        }
        inline float operator()(const float a) const {
            return a > scalar;
        }
    };

    class AddScalar {
    private:
        const float scalar;
    public:
        AddScalar(const float _scalar) : scalar(_scalar) {
        }
        inline float operator()(const float a) const {
            return a + scalar;
        }
    };

    class WeightedAddScalar {
    private:
        const float weight, scalar;
    public:
        WeightedAddScalar(const float _weight, const float _scalar) : weight(_weight), scalar(_scalar) {
        }
        inline float operator()(const float a) const {
            return weight * a + scalar;
        }
    };

    class MultByScalar {
    private:
        const float scalar;
    public:
        MultByScalar(const float _scalar) : scalar(_scalar) {
        }
        inline float operator()(const float a) const {
            return a * scalar;
        }
    };

    class Pow {
    private:
        const float p;
    public:
        Pow(const float _p) : p(_p) {
        }
        inline float operator()(const float a) const {
            return _pow(a, p);
        }
    };

    template <bool exclusive>
    class InRange {
    private:
        const float lower, upper;
    public:
        InRange(const float _lower, const float _upper) : lower(_lower), upper(_upper) {
        }
        inline float operator()(const float a) const {
            return exclusive ? a > lower && a < upper : a >= lower && a <= upper;
        }
    };

    class MinWithScalar {
    private:
        const float scalar;
    public:
        MinWithScalar(const float _scalar) : scalar(_scalar) {
        }
        inline float operator()(const float a) const {
            return a > scalar ? scalar : a;
        }
    };

    class MaxWithScalar {
    private:
        const float scalar;
    public:
        MaxWithScalar(const float _scalar) : scalar(_scalar) {
        }
        inline float operator()(const float a) const {
            return a > scalar ? a : scalar;
        }
    };
};

class MatrixBinaryOps {
public:
    class Equals {
    public:
        inline float operator()(const float a, const float b) const {
            return a == b;
        }
    };

    class BiggerThan {
    public:
        inline float operator()(const float a, const float b) const {
            return a > b;
        }
    };

    class Divide {
    public:
        inline float operator()(const float a, const float b) const  {
            return a / b;
        }
    };

    class Multiply {
    public:
        inline float operator()(const float a, const float b) const {
            return a * b;
        }
    };

    class SquaredDiff {
    public:
        inline float operator()(const float a, const float b) const {
            return (a - b) * (a - b);
        }
    };

    class WeightedAdd {
    private:
        const float scaleA, scaleB;
    public:
        WeightedAdd(const float _scaleA, const float _scaleB) : scaleA(_scaleA), scaleB(_scaleB) {
        }
        inline float operator()(const float a, const float b) const {
            return a * scaleA + b * scaleB;
        }
    };

    class Add {
    public:
        inline float operator()(const float a, const float b) const {
            return a + b;
        }
    };
    
    class First {
    public:
        inline float operator()(const float a, const float b) const {
            return a;
        }
    };
    
    class Second {
    public:
        inline float operator()(const float a, const float b) const {
            return b;
        }
    };
    
    class SecondScaled {
    private:
        const float scale;
    public:
        SecondScaled(const float _scale) : scale(_scale) {
        }
        inline float operator()(const float a, const float b) const {
            return scale * b;
        }
    };
};

class MatrixAggs {
public:
    class Sum {
    public:
        inline float operator()(const float a, const float b) const {
            return a + b;
        }
        inline float getBaseValue() {
            return 0;
        }
    };

    class Max {
    public:
        inline float operator()(const float a, const float b) const {
            return a > b ? a : b;
        }
        inline float getBaseValue() {
            return -2e38;
        }
    };

    class Min {
    public:
        inline float operator()(const float a, const float b) const {
            return a > b ? b : a;
        }
        inline float getBaseValue() {
            return 2e38;
        }
    };

};


#endif	/* NVMATRIX_OPERATORS_H_ */

