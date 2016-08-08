//
// Created by jiajaichen on 16-8-5.
//

#include "window_caffe_ssd_detector.h"

#include "vehicle_caffe_detector.h"
#include "alg/caffe_helper.h"

namespace dg {
WindowCaffeSsdDetector::WindowCaffeSsdDetector(const VehicleCaffeDetectorConfig &config) : caffe_config_(config) {

    use_gpu_ = config.use_gpu;
    gpu_id_ = config.gpu_id;
    threshold_ = config.threshold;

    if (use_gpu_) {

        Caffe::SetDevice(gpu_id_);
        Caffe::set_mode(Caffe::GPU);

        use_gpu_ = true;
    }
    else {
        Caffe::set_mode(Caffe::CPU);
        use_gpu_ = false;

    }

    batch_size_ = config.batch_size;
    //  net_.reset(new Net<float>(config.deploy_file, TEST));

#if DEBUG
    net_.reset(
        new Net<float>(config.deploy_file, TEST));
#else
    net_.reset(
            new Net<float>(config.deploy_file, TEST, config.is_model_encrypt));
#endif

    net_->CopyTrainedLayersFrom(config.model_file);

    Blob<float> *input_layer = net_->input_blobs()[0];
    num_channels_ = input_layer->channels();
    input_geometry_ = cv::Size(input_layer->width(),input_layer->height());
    target_col_=config.target_min_size;
    target_row_=config.target_max_size;

  /*  input_layer->Reshape(batch_size_, num_channels_,
                         input_geometry_.height,
                         input_geometry_.width);
    net_->Reshape();*/

/*    const vector<boost::shared_ptr<Layer<float> > > &layers = net_->layers();
    const vector<vector<Blob<float> *> > &bottom_vecs = net_->bottom_vecs();
    const vector<vector<Blob<float> *> > &top_vecs = net_->top_vecs();
    for (int i = 0; i < layers.size(); ++i) {
        layers[i]->Forward(bottom_vecs[i], top_vecs[i]);
    }
*/
    device_setted_ = false;

}

WindowCaffeSsdDetector::~WindowCaffeSsdDetector() {

}
void WindowCaffeSsdDetector::Fullfil(vector<cv::Mat> &images_origin, vector<Blob < float> *>&outputs_win,vector<vector<Detection> > &detect_results) {
    int image_offset = detect_results.size();
    for (int i = 0; i < images_origin.size(); ++i) {
        vector<Detection> imageDetection;
        detect_results.push_back(imageDetection);
    }
    int tot_cnt_win = 0;
    int box_num_win = outputs_win[tot_cnt_win]->height();
    const float* top_data_win = outputs_win[tot_cnt_win]->cpu_data();

    for(int j = 0; j < box_num_win; j++) {

        int img_id = top_data_win[j * 7 + 0];

        if ((img_id < 0) || ((img_id+image_offset) >= detect_results.size())) {
            LOG(ERROR) << "Image id invalid: " << img_id << endl;
            continue;
        }
        vector<Detection> &imageDetection = detect_results[image_offset + img_id];
        if(imageDetection.size()>0){
            continue;
        }
//        int cls = top_data_win[j * 7 + 1];
        float score = top_data_win[j * 7 + 2];
        float xmin = top_data_win[j * 7 + 3] * images_origin[img_id].cols;
        float ymin = top_data_win[j * 7 + 4] * images_origin[img_id].rows;
        float xmax = top_data_win[j * 7 + 5] * images_origin[img_id].cols;
        float ymax = top_data_win[j * 7 + 6] * images_origin[img_id].rows;
        if (score > threshold_) {
            /*******************tiny object detector*********************/
            Detection detection;
            detection.box = Rect(xmin, ymin, xmax - xmin, ymax - ymin);
            detection.id = img_id;
            detection.confidence = score;
     
            imageDetection.push_back(detection);


        }
    }
}
int WindowCaffeSsdDetector::DetectBatch(vector<cv::Mat> &img,
                                      vector<vector<Detection> > &detect_results) {

    if (!device_setted_) {
        Caffe::SetDevice(gpu_id_);
        Caffe::set_mode(Caffe::GPU);
        device_setted_ = true;
    }


    detect_results.clear();
    vector<cv::Mat> toPredict;
    vector<cv::Mat> origins;
    for (int i = 0; i < img.size(); ++i) {
        cv::Mat image = img[i].clone();
        resize(image,image,Size(target_col_,target_row_));
        toPredict.push_back(image);
        origins.push_back(img[i]);
        if (toPredict.size() == batch_size_) {
            vector<Blob<float> *> outputs = PredictBatch(toPredict);
            Fullfil(origins, outputs, detect_results);
            toPredict.clear();
            origins.clear();
        }
    }

    if (toPredict.size() > 0) {
        vector<Blob<float> *> outputs = PredictBatch(toPredict);

        Fullfil(origins, outputs, detect_results);
    }

}

std::vector<Blob<float> *> WindowCaffeSsdDetector::PredictBatch(const vector<Mat> &imgs) {

    vector<Blob<float> *> outputs;

    Blob<float> *input_layer = net_->input_blobs()[0];
    if (imgs.size() <= batch_size_) {
        input_layer->Reshape(imgs.size(), num_channels_,
                             input_geometry_.height,
                             input_geometry_.width);
        net_->Reshape();
    } else {
        LOG(ERROR) << "Input images size is more than batch size!" << endl;
        return outputs;
    }
    float *input_data = input_layer->mutable_cpu_data();
    int cnt = 0;
    DLOG(INFO) << "Start predict batch, size: " << imgs.size() << endl;
    for (int i = 0; i < imgs.size(); i++) {
        cv::Mat sample;
        cv::Mat img = imgs[i];

        GenerateSample(num_channels_, img, sample);
        if ((sample.rows != input_geometry_.height) || (sample.cols != input_geometry_.width)) {
            cv::resize(sample, sample, Size(input_geometry_.width, input_geometry_.height));
        }
        float mean[3] = {104, 117, 123};
        for (int k = 0; k < sample.channels(); k++) {
            for (int i = 0; i < sample.rows; i++) {
                for (int j = 0; j < sample.cols; j++) {
                    input_data[cnt] = sample.at<uchar>(i, j * 3 + k) - mean[k];
                    cnt += 1;
                }
            }
        }
    }

    net_->ForwardPrefilled();

    if (use_gpu_) {
        cudaDeviceSynchronize();
    }

    for (int i = 0; i < net_->num_outputs(); i++) {
        Blob<float> *output_layer = net_->output_blobs()[i];
        outputs.push_back(output_layer);
    }

    return outputs;
}

}