
#ifndef NEURONS_H_
#define	NEURONS_H_

#include <assert.h>
#include <string>
#include "matrix.h"
#include "util.h"

using namespace std;
/* =======================
 * Neuron
 * -----------------------
 * 
 * f(x) = x
 * =======================
 */
class Neuron {
protected:
    virtual void _activate(Matrix *_inputs, Matrix *_outputs) {
        if (_inputs != _outputs) {
            _inputs->copy(*_outputs);
        }
    }
    virtual void _activate(Matrix *_inputs) {
    }
public:
    Neuron() {
    }
    virtual void activate(Matrix& inputs, Matrix& outputs) {
        _activate(&inputs, &outputs);
    }
    
    virtual void activate(Matrix & inputs) {
        _activate(&inputs);
    }

    static Neuron& makeNeuron(dictParam_t &paramsDict);
    static Neuron& makeNeuron(dictParam_t &paramsDict, const string &type);
};

/* =======================
 * LogisticNeuron
 * -----------------------
 * 
 * f(x) = 1 / (1 + e^-x)
 * =======================
 */
class LogisticNeuron : public Neuron {
protected:
    void _activate(Matrix *_inputs, Matrix *_outputs) {
        _inputs->apply(Logistic(), *_outputs);
    }

    virtual void _activate(Matrix *_inputs) {
        _inputs->apply(Logistic());
    }
public:
    class Logistic {
    public:
        inline float operator()(const float a) const {
            return 1.0f / (1.0f + _exp(-a));
        }
    };
    
    LogisticNeuron() : Neuron() {
    }
};

/* =======================
 * RampNeuron (from YOLO paper)
 * -----------------------
 * 
 * f(x) = (x < 0) ? (0.1 * x) : (1.1 * x)
 * =======================
*/
class RampNeuron: public Neuron {
protected:
    void _activate(Matrix *_inputs, Matrix *_outputs) {
        _inputs->apply(RampOperator(), *_outputs);
    }

    virtual void _activate(Matrix *_inputs) {
        _inputs->apply(RampOperator());
    }
public:
    class RampOperator {
    public:    
        inline float operator()(float x) const {
            return x < 0.0f ? (0.1 * x) : (1.1 * x);
        }
    };

    RampNeuron() : Neuron() {
    }
};

/* =======================
 * ReluNeuron
 * -----------------------
 * 
 * f(x) = max(0, x)
 * =======================
 */
class ReluNeuron : public Neuron {
protected:
    void _activate(Matrix *_inputs, Matrix *_outputs) {
        _inputs->apply(ReluOperator(), *_outputs);
    }

    virtual void _activate(Matrix *_inputs) {
        _inputs->apply(ReluOperator());
    }
public:
    class ReluOperator {
    public:    
        inline float operator()(float x) const {
            return x < 0.0f ? 0.0f : x;
        }
    };

    ReluNeuron() : Neuron() {
    }
};

/* =======================
 * BoundedReluNeuron
 * -----------------------
 * 
 * f(x) = min(a, max(0, x))
 * =======================
 */
class BoundedReluNeuron : public Neuron {
protected:
    float _a;
    
    void _activate(Matrix *_inputs, Matrix *_outputs) {
        _inputs->apply(BoundedReluOperator(_a), *_outputs);
    }

public:
    class BoundedReluOperator {
    private:
        float _a;
    public:
        BoundedReluOperator(float a) : _a(a) {
        }
        inline float operator()(float x) const {
            return x < 0.0f ? 0.0f : x > _a ? _a : x;
        }
    };
    
    BoundedReluNeuron(float a) : Neuron(), _a(a) {
    }
};


/* =======================
 * TanhNeuron
 * -----------------------
 * 
 * f(x) = a*tanh(b*x)
 * =======================
 */
class TanhNeuron : public Neuron {
protected:
    float _a, _b;

    void _activate(Matrix *_inputs, Matrix *_outputs) {
        _inputs->apply(TanhOperator(_a, _b), *_outputs);
    }

    virtual void _activate(Matrix *_inputs) {
        _inputs->apply(TanhOperator(_a, _b));
    }
public:
    class TanhOperator {
    private:
        float _a, _n2b;
    public:
        TanhOperator(float a, float b) : _a(a), _n2b(-2*b) {
        }
        virtual inline float operator()(float x) const {
            return _a * (2.0f / (1.0f + _exp(x * _n2b)) - 1.0f);
        }
    };

    
    TanhNeuron(float a, float b) : Neuron(), _a(a), _b(b) {
    }
};

/* =======================
 * SoftReluNeuron
 * -----------------------
 * 
 * f(x) = log(1 + e^x)
 * =======================
 */
class SoftReluNeuron : public Neuron {
protected:
    void _activate(Matrix *_inputs, Matrix *_outputs) {
        assert(_inputs != _outputs);
        _inputs->apply(SoftReluOperator(), *_outputs);
    }

    virtual void _activate(Matrix *_inputs) {
        _inputs->apply(SoftReluOperator());
    }
public:
    class SoftReluOperator {
    public:    
        inline float operator()(float x) const {
            // This piece-wise implementation has better numerical stability than
            // simply computing log(1 + e^x).
            return x > 4.0f ? x : _log(1.0f + _exp(x));
        }
    };
    
    SoftReluNeuron() : Neuron() {
    }
};

/* =======================
 * SquareNeuron
 * -----------------------
 * 
 * f(x) = x^2
 * =======================
 */
class SquareNeuron : public Neuron {
protected:
    void _activate(Matrix *_inputs, Matrix *_outputs) {
        assert(_inputs != _outputs);
        _inputs->apply(Square(), *_outputs);
    }

    void _activate(Matrix *_inputs) {
        _inputs->apply(Square());
    }
public:
    class Square {
    public:
        inline float operator()(const float a) const {
            return a * a;
        }
    };
    
    SquareNeuron() : Neuron() {
    }
};

/* =======================
 * SqrtNeuron
 * -----------------------
 * 
 * f(x) = sqrt(x)
 * =======================
 */
class SqrtNeuron : public Neuron {
protected:
    void _activate(Matrix *_inputs, Matrix *_outputs) {
        _inputs->apply(Sqrt(), *_outputs);
    }

    void _activate(Matrix *_inputs) {
        _inputs->apply(Sqrt());
    }
public:
    class Sqrt {
    public:
        inline float operator()(const float a) const {
            return sqrt(a);
        }
    };
    
    SqrtNeuron() : Neuron() {
    }
};

/* =======================
 * LinearNeuron
 * -----------------------
 * 
 * f(x) = a*x + b
 * =======================
 */
class LinearNeuron : public Neuron {
protected:
    float _a, _b;
    void _activate(Matrix *_inputs, Matrix *_outputs) {
        _inputs->apply(LinearOperator(_a, _b), *_outputs);
    }

    void _activate(Matrix *_inputs) {
        _inputs->apply(LinearOperator(_a, _b));
    }
public:
    class LinearOperator {
    protected:
        float _a, _b;
    public:    
        inline float operator()(float x) const {
            return _a * x + _b;
        }
        LinearOperator(float a, float b) : _a(a), _b(b) {
        }
    };
    
    LinearNeuron(float a, float b) : Neuron(), _a(a), _b(b) {
    }
};
#endif	/* NEURONS_H_ */

