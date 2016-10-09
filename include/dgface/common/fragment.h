
#ifndef FRAGMENT_H_
#define FRAGMENT_H_
#include <string>
#include <vector>
#include <map>

using namespace std;

/*
 * Fragment class for shared convolution and max-pooling
 */
class Fragment {
protected:
    int _sizeX, _sizeY, _mapNum;
    Matrix *_featureMaps;
public:
    Fragment(int sizeX, int sizeY, int mapNum):
        _sizeX(sizeX), _sizeY(sizeY), _mapNum(mapNum) {
            if (sizeX > 0 && sizeY > 0 && mapNum > 0) {
                _featureMaps = new Matrix(sizeX * sizeY, mapNum);
            } else {
                _featureMaps = NULL;
            }
    }
    Fragment(float *data, int sizeX, int sizeY, int mapNum):
        _sizeX(sizeX), _sizeY(sizeY), _mapNum(mapNum) {
            if (sizeX > 0 && sizeY > 0 && mapNum > 0) {
                _featureMaps = new Matrix(data, sizeX * sizeY, mapNum, false, true);
            } else {
                _featureMaps = NULL;
            }
    }
    virtual ~Fragment() {
        if (_featureMaps) {
            delete _featureMaps;
            _featureMaps = NULL;
        }
    }

    int getSizeX() {
        return _sizeX;
    }
    int getSizeY() {
        return _sizeY;
    }
    int getMapNum() {
        return _mapNum;
    }

    Matrix *getMaps() {
        return _featureMaps;
    }
};

typedef std::vector<Fragment *> FragmentV;
typedef std::map<string, FragmentV*> FragmentVM;

#endif
