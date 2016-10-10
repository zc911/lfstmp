/**
 * layer.h
 *
 * Author: dudalong (dudalong@baidu.com)
 * Created on: 2013-03-16
 *
 * Copyright (c) Baidu.com, Inc. All Rights Reserved
 *
 */
#ifndef LAYER_H_
#define	LAYER_H_

#include <string>
#include <vector>
#include <map>
#include <assert.h>
#include <matrix.h>
#include <iostream>
#include "convnet.h"
#include "weights.h"
#include "neuron.h"
#include "util.h"
#include "matrix_ssemul.h"
#include "fragment.h"

using namespace std;

class ConvNet;
/*
 * Abstract layer.
 */
class Layer {
protected:
    ConvNet* _convNet;
    std::vector<Layer*> _prev, _next;
    
    std::string _name, _type;
    virtual void fpropActs(Matrix &input, int inpIdx, float scaleTargets, Matrix &output);
    virtual void fpropActsFragment(FragmentV &fragments, int inpIdx, float scaleTargets, FragmentV &fragOutput);
public:
    Layer(ConvNet *convNet, string name, string type);
    Layer(ConvNet *convNet, dictParam_t &paramsDict);
    virtual ~Layer();
    virtual void fpropActs(map<string, Matrix*> &outputMap);
    virtual void fprop_acts_caffe(map<string, Matrix*> &outputMap);
    void fprop(Matrix &data, MatrixM &prevOut);
    void fprop(Matrix &data, Matrix &output, vector<string> &outlayer, MatrixM &results);
    void fpropFragment(FragmentV &data, int strideX, int strideY, FragmentVM &convPrevOut,
            MatrixM &fcPrevOut, bool &isConvFlag);
    std::string& getName();
    std::string& getType();
    void addNext(Layer* l);
    void addPrev(Layer* l);
    std::vector<Layer*>& getPrev();
    std::vector<Layer*>& getNext();
    virtual int getOutputsX() {return 0;};
    virtual int getLabelsDim(){return 0;};
    virtual int getDataDim(){return 0;};
    virtual int get_input_channel(){
        return 0;
    };//add by yuzhuo 1210
    Matrix &getOutputVal(map<string, Matrix*> &outputMap) {
        return *outputMap[_name];
    }
    Matrix &getInputVal(int i, map<string, Matrix*>& outputMap) {
        return _prev[i]->getOutputVal(outputMap);
    }
};

class NeuronLayer : public Layer {
protected:
    Neuron* _neuron;

    void fpropActs(Matrix &input, int inpIdx, float scaleTargets, Matrix &output);
    void fpropActsFragment(FragmentV &fragments, int inpIdx, float scaleTargets, FragmentV &fragOutput);
public:
    NeuronLayer(ConvNet *convNet, dictParam_t &paramsDict);
    ~NeuronLayer();
    void fpropActs(map<string, Matrix*> &outputMap);
};

class OutLayer : public Layer {
    void fpropActs(Matrix &input, int inpIdx, float scaleTargets, Matrix &output);
    void fpropActsFragment(FragmentV &fragments, int inpIdx, float scaleTargets, FragmentV &fragOutput);
public:
    OutLayer(ConvNet *convNet, string name, string type);
    int getLabelsDim() {
        int dim = 0;
        for (int i = 0; i < _prev.size(); i++) {
	    dim += _prev[i]->getLabelsDim();
        }
        return dim;
    }
    void fpropActs(map<string, Matrix*> &outputMap);
};


class WeightLayer : public Layer {
protected:
    WeightList _weights;
    Weights *_biases;
public:
    WeightLayer(ConvNet *convNet, dictParam_t &paramsDict);
    Weights& getWeights(int idx);
    virtual ~WeightLayer() {
        if (_biases != NULL) {
            delete _biases;
        }
    }
};

class FCLayer : public WeightLayer {
protected:
    int _sparseFlag;
    csc_t **_cscMat;
    void fpropActs(Matrix &input, int inpIdx, float scaleTargets, Matrix &output);
public:
    FCLayer(ConvNet *convNet, dictParam_t &paramsDict);
    int getLabelsDim() {
	return (*_weights[0]).getNumCols();
    }
    ~FCLayer();
    void fpropActs(map<string, Matrix*> &outputMap);
};

class SoftmaxLayer : public Layer {
protected:
    int _labelsDim;
    void fpropActs(Matrix &input, int inpIdx, float scaleTargets, Matrix &output);
public:
    SoftmaxLayer(ConvNet *convNet, dictParam_t &paramsDict);
    int getLabelsDim() {
        return _labelsDim;
    }
    void fpropActs(map<string, Matrix*> &outputMap);
};

//for metric learning, by xiatian, 2014.11.13
class HybridReluTanhLayer : public Layer {
protected:
    int _labelsDim;
    void fpropActs(Matrix &input, int inpIdx, float scaleTargets, Matrix &output);
public:
    HybridReluTanhLayer(ConvNet *convNet, dictParam_t &paramsDict);
    int getLabelsDim() {
	return _labelsDim;
    }
    void fpropActs(map<string, Matrix*> &outputMap);
};

class DataLayer : public Layer {
protected:
    int _inputDim;
    int _offset;
    void fpropActs(Matrix &input, int inpIdx, float scaleTargets, Matrix &output);
    void fpropActs(Matrix &input);
public:
    DataLayer(ConvNet *convNet, dictParam_t &paramsDict, int isMultiPatch=0);
    int getDataDim() {
        return _inputDim;
    }
    int getDataOffset() {
        return _offset;
    }
    void fpropActs(map<string, Matrix*> &outputMap);

};

class LocalLayer : public WeightLayer {
protected:
    intv* _padding, *_stride, *_filterSize, *_channels, *_imgSize, *_groups, *_filterChannels;
    int _modulesX, _modules, _numFilters;
    int **imgOffsetOut, **imgOffsetIn;
    void makeOffset();
    void fpropActs(Matrix &input, int inpIdx, float scaleTargets, Matrix &output);

public:
    LocalLayer(ConvNet *convNet, dictParam_t &paramsDict);
    ~LocalLayer();
    int getImgSize(int idx) {
        assert(idx < _imgSize->size());
        return _imgSize->at(idx);
    }
    int getOutputsX() {
        return _modulesX;
    }
    int getStride(int idx) {
        assert(idx < _stride->size());
        return _stride->at(idx);
    }
    int get_padding(int idx) {
        assert(idx < _padding->size());
        return _padding->at(idx);
    }
    int get_input_channel() {
        return (_channels)->at(0); //add by yuzhuo 1210
    }
};

class ConvLayer : public LocalLayer {
protected:
    bool _sharedBiases;
    void fpropActs(Matrix &input, int inpIdx, float scaleTargets, Matrix &output);
    void fpropActsFragment(FragmentV &fragments, int inpIdx, float scaleTargets, FragmentV &fragOutput);
public:
    ConvLayer(ConvNet *convNet, dictParam_t &paramsDict);
    ~ConvLayer(){};
    void fpropActs(map<string, Matrix*> &outputMap);
    void fprop_acts_caffe(map<string, Matrix*> &outputMap);
};

#define fprop_acts fpropActs
class DepthConcatLayer : public Layer {
protected:
    intv* _depth;
    void fprop_acts(Matrix &input, int inpIdx, float scaleTargets, Matrix &output);
public:
    DepthConcatLayer(ConvNet *convNet, dictParam_t &paramsDict);
    ~DepthConcatLayer();
    void fprop_acts(map<string, Matrix*> &outputMap);
};

class DeConvLayer : public LocalLayer {
protected:
    bool _sharedBiases;
    void fpropActs(Matrix &input, int inpIdx, float scaleTargets, Matrix &output);
public:
    DeConvLayer(ConvNet *convNet, dictParam_t &paramsDict);
    ~DeConvLayer(){};
};

class PoolLayer : public Layer {
protected:
    int _channels, _sizeX, _start, _stride, _outputsX;
    int _imgSize;
    string _pool;
public:
    PoolLayer(ConvNet *convNet, dictParam_t &paramsDict);
    static PoolLayer& makePoolLayer(ConvNet* convNet, dictParam_t &paramsDict);

    int getOutputsX() {
        return _outputsX;
    }
    int getStride() {
        return _stride;
    }
};

class AvgPoolLayer : public PoolLayer {
protected:
    void fpropActs(Matrix &input, int inpIdx, float scaleTargets, Matrix &output);
    void fprop_acts_caffe(map<string, Matrix*> &outputMap);
public:
    AvgPoolLayer(ConvNet *convNet, dictParam_t &paramsDict);
};

class MaxPoolLayer : public PoolLayer {
protected:
    void fpropActs(Matrix &input, int inpIdx, float scaleTargets, Matrix &output);
    void fpropActsFragment(FragmentV &fragments, int inpIdx, float scaleTargets, FragmentV &fragOutput);
public:
    MaxPoolLayer(ConvNet *convNet, dictParam_t &paramsDict);
    void fpropActs(map<string, Matrix*> &outputMap);
    void fprop_acts_caffe(map<string, Matrix*> &outputMap);
};

//==================ROI LAYER========
class ROIPoolLayer : public Layer {
protected:
    int _poolw;
    int _poolh;
    int _channels;
    int _start;
    int _outputsx;
    int _imgsize;
    float _spatial_scale;
    Matrix* _tmpmat;
    void fprop_acts_caffe(Matrix &input, int inpIdx, float scaleTargets, Matrix &output);
    void fprop_acts_caffe(map<string, Matrix*> &outputMap);

public:
    ROIPoolLayer(ConvNet *convnet, dictParam_t &paramsDict);
    ~ROIPoolLayer(){};
    int get_outputsx(){
        return _outputsx;
    }
};

class ResponseNormLayer : public Layer {
protected:
    int _channels, _size;
    float _scale, _pow;
    float _kconst;

    void fpropActs(Matrix &input, int inpIdx, float scaleTargets, Matrix &output);
public:
    ResponseNormLayer(ConvNet *convNet, dictParam_t &paramsDict);
}; 

class CrossMapResponseNormLayer : public ResponseNormLayer {
protected:
    int _imgSize;
    void fpropActs(Matrix &input, int inpIdx, float scaleTargets, Matrix &output);
    void fpropActsFragment(FragmentV &fragments, int inpIdx, float scaleTargets, FragmentV &fragOutput);
public:
    CrossMapResponseNormLayer(ConvNet *convNet, dictParam_t &paramsDict);
    int getOutputsX() {
        return _imgSize;
    }
    void fpropActs(map<string, Matrix*> &outputMap);
    void fprop_acts_caffe(map<string, Matrix*> &outputMap);
}; 

class ContrastNormLayer : public ResponseNormLayer {
protected:
    int _imgSize;
    void fpropActs(Matrix &input, int inpIdx, float scaleTargets, Matrix &output);
public:
    ContrastNormLayer(ConvNet *convNet, dictParam_t &paramsDict);
    int getOutputsX() {
        return _imgSize;
    }
};

class BlockExpandLayer : public Layer {
protected:
    int _blockSizeX, _blockSizeY, _strideX, _strideY, _paddingX, _paddingY;
    int _channels;
    void getBlockOutNum(Matrix &input, int &blockOutputX, int &blockOutputY);
public:
    BlockExpandLayer(ConvNet *convnet, dictParam_t &paramsDict);
    ~BlockExpandLayer();
    void fpropActs(map<string, Matrix*> &outputMap);
};

class BLstmLayer : public Layer {
protected:
    int /*_cellsPerBlock,*/ _numBlocks, _reversed;
    Matrix *_recurrBias, *_recurrWeight;
    Matrix *_peepIG, *_peepOG, *_peepFG;
    string _inputNeuronType, _stateNeuronType, _gateNeuronType;
    Neuron *_inputNeuron, *_gateNeuron, *_stateNeuron;
public:
    BLstmLayer(ConvNet *convnet, dictParam_t &paramsDict);
    ~BLstmLayer();
    void fpropActs(map<string, Matrix*> &outputMap);
};

class GatedRNNLayer : public Layer {
protected:
    int /*_cellsPerBlock,*/ _num_blocks, _reversed;
    Matrix *_gated_recurr_bias, *_state_weight, *_gate_weight;
    string _state_neuron_type, _gate_neuron_type;
    Neuron *_gate_neuron, *_state_neuron;
public:
    GatedRNNLayer(ConvNet *convnet, dictParam_t &paramsDict);
    ~GatedRNNLayer();
    void fprop_acts(map<string, Matrix*> &outputMap);
};

class MaxoutLayer : public Layer {
protected:
    int _channels;
    int _groups;
    void fprop_acts(map<string, Matrix*> &outputMap);
public:
    MaxoutLayer(ConvNet *convnet, dictParam_t &paramsDict);
    ~MaxoutLayer(){};
};

#endif	/* LAYER_H_ */

