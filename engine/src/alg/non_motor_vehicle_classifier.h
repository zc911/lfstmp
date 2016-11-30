
#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "caffe/caffe.hpp"
#include "util/caffe_helper.h"

using namespace std;
using namespace cv;
using namespace caffe;

#ifndef NON_MOTOR_VEHICLE_CLASSIFIER_H_
#define NON_MOTOR_VEHICLE_CLASSIFIER_H_

namespace dg {

class Faster_rcnn {
public:
    Faster_rcnn(const string& model_file,
            const string& trained_file,
            const string& layer_name_cls,
            const string& layer_name_reg,
            const bool use_GPU,
            const Size &image_size,
            const float conf_thres,
            const int max_per_img,
            const int sliding_window_stride,
            const vector<float> &area,
            const vector<float> &ratio);
    void edge_complete(const vector<Mat> &imgs, vector<Mat> &imgs_new, const Size &image_size);
    void forward(const vector<Mat> &imgs, vector<Blob<float>* > &outputs);
    void get_detection(vector<Blob<float>* >& outputs, vector< vector<struct Bbox> > &final_vbbox);

private:
    void nms_zz(vector<struct Bbox>& p, float threshold, int num_thresh);
    void nms(vector<struct Bbox>& p, float threshold);
    caffe::shared_ptr<Net<float> > net_;
    int num_channels_;
    int batch_size_;
    bool useGPU_;
    vector<float> pixel_means_;
    float  conf_thres_;
    Size   image_size_;
    string layer_name_cls_;
    string layer_name_reg_;
    int sliding_window_stride_;
    int max_per_img_;
    vector<float> area_;
    vector<float> ratio_;
};

class CaffeAttribute {
    public:
        typedef struct {
            string name;
            float thresh_low;
            float thresh_high;
            int idx;
            float confidence;
            int mappingId;
            int categoryId;
        }Attrib;
        CaffeAttribute(const string& attrib_table_path,
                const string& model_file,
                const string& trained_file,
                const string& layer_name,
                const int height,
                const int width,
                const int crop_height,
                const int crop_width,
                const int pixel_scale,
                const bool use_GPU = 1);
        void BatchAttributePredict(const vector<Mat> &imgs, vector<vector<float> > &results);
        void AttributePredict(const vector<Mat> &imgs, vector<vector<float> > &results);
        vector<Attrib> _attrib_table;
        int _batch_size;
    private:
        void load_names(const string &name_list, vector<Attrib> &attribs);
        caffe::shared_ptr<Net<float> > _net;
        Rect _crop_rect;
        bool _useGPU; 
        string _layer_name;
        int _num_channels;
        int _height;
        int _width;
        int _crop_height;
        int _crop_width;
        int _pixel_scale;
        float _pixel_means[3];
};

class NonMotorVehicleClassifier {
public:
    typedef struct {
        bool is_model_encrypt = true;
        int gpu_id = 0;
        bool use_gpu = 0;
		string bitri_trained_file;
		string bitri_deploy_file;
		string upper_trained_file;
		string upper_deploy_file;
		string rpn_trained_file;
		string rpn_deploy_file;
		string attrib_table_path;
		string bitri_layer_name;
		string upper_layer_name;
    } NonMotorVehicleConfig;

	NonMotorVehicleClassifier (NonMotorVehicleConfig & nonMotorVehicleConfig);
	~NonMotorVehicleClassifier();

	void BatchClassify(const vector<cv::Mat> &imgs, vector<vector<CaffeAttribute::Attrib> > &results);

private:
    void process_batch_after_det(CaffeAttribute &upper_attrib, 
        const vector<Mat> &images_ori, 
        const vector< vector<Rect> > &bboxes_list, 
        vector<vector<float> > &results, 
        size_t batch_size);
	NonMotorVehicleConfig config;
	string layer_name_cls = "conv_face_16_cls";
	string layer_name_reg = "conv_face_16_reg";
	Size image_size = Size(300, 300);
	float det_thresh = 0.9;
	int max_per_img = 2;
	int sliding_window_stride = 16;
	vector<float> area  = {48 * 48, 96 * 96, 2 * 96 * 96, 192 * 192};
	vector<float> ratio = {1};

};


} /* namespace dg */

#endif