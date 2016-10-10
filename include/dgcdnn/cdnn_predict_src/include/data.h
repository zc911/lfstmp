
#ifndef DATA_H_
#define	DATA_H_

#include <vector>
#include <algorithm>
#include "matrix.h"

class DataProvider {
protected:
    Matrix* _hData;
    
    int _minibatchSize;
    long int _dataSize;
public:
    DataProvider(int minibatchSize);
    Matrix& operator[](int idx);
    void setData(Matrix&);
    void clearData();
    Matrix& getMinibatch(int idx);
    Matrix& getDataSlice(int startCase, int endCase);
    int getNumMinibatches();
    int getMinibatchSize();
    int getNumCases();
    int getNumCasesInMinibatch(int idx);
};

#endif	/* DATA_H_ */

