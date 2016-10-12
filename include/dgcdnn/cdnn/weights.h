
#ifndef WEIGHTS_H_
#define	WEIGHTS_H_

#include <string>
#include <vector>
#include "matrix.h"

using namespace std;

class Weights {
private:
    Matrix* _weights;
    
public:
    Matrix& operator*() {
        return getW();
    }
    
    Weights(Matrix& hWeights)
        : _weights(&hWeights) {
    }
        
    ~Weights() {
        delete _weights;
    }
    
    Matrix& getW() {
        return *_weights;
    }
    
    int getNumRows() const {
        return _weights->getNumRows();
    }
    
    int getNumCols() const {
        return _weights->getNumCols();
    }
 
};

class WeightList {
private:
    std::vector<Weights*> _weightList;

public:
    Weights& operator[](const int idx) const {
        return *_weightList[idx];
    }
    
    ~WeightList() {
        for (unsigned int i = 0; i < _weightList.size(); i++) {
            delete _weightList[i];
        }
    }

    WeightList() {
    }

    void addWeights(Weights& w) {
        _weightList.push_back(&w);
    }

    int getSize() {
        return _weightList.size();
    }
};

#endif	/* WEIGHTS_H_ */
