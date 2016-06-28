//
// Created by chenzhen on 5/24/16.
//

#include <ftadvanc.h>
#include "vehicle_caffe_detector.h"
#include "caffe_helper.h"

namespace dg {
VehicleCaffeDetector::VehicleCaffeDetector(const VehicleCaffeDetectorConfig &config) : caffe_config_(config) {

    use_gpu_ = config.use_gpu;
    gpu_id_ = config.gpu_id;
    threshold_ = config.threshold;
    if (use_gpu_) {
        Caffe::set_mode(Caffe::GPU);
        Caffe::SetDevice(gpu_id_);
        use_gpu_ = true;
    }
    else {
        Caffe::set_mode(Caffe::CPU);
        use_gpu_ = false;

    }

    batch_size_ = config.batch_size;
  //  net_.reset(new Net<float>(config.deploy_file, TEST));

    net_.reset(
             new Net<float>(config.deploy_file, TEST, config.is_model_encrypt, NULL));
    net_->CopyTrainedLayersFrom(config.model_file);
    cout<<config.model_file<<endl;

    Blob<float> *input_layer = net_->input_blobs()[0];
    num_channels_ = input_layer->channels();
    input_geometry_ = cv::Size(input_layer->width(), input_layer->height());
    input_layer->Reshape(batch_size_, num_channels_,
                         input_geometry_.height,
                         input_geometry_.width);
    net_->Reshape();
  /*  const vector<boost::shared_ptr<Layer<float> > > & layers = net_->layers();
    const vector<vector<Blob<float>* > > & bottom_vecs = net_->bottom_vecs();
    const vector<vector<Blob<float>* > > & top_vecs = net_->top_vecs();
    for(int i = 0; i < layers.size(); ++i) {
        layers[i]->Forward(bottom_vecs[i], top_vecs[i]);
    }
*/
    device_setted_ = false;
#ifdef SHOW_VIS
    color_.push_back(Scalar(255, 0, 0));
    color_.push_back(Scalar(0, 255, 0));
    color_.push_back(Scalar(0, 0, 255));
    color_.push_back(Scalar(255, 255, 0));
    color_.push_back(Scalar(0, 255, 255));
    color_.push_back(Scalar(255, 0, 255));
    tags_.push_back("bg");
    tags_.push_back("car");
    tags_.push_back("person");
    tags_.push_back("bicycle");
    tags_.push_back("tricycle");
#endif

}

VehicleCaffeDetector::~VehicleCaffeDetector() {

}

void VehicleCaffeDetector::Fullfil(vector<cv::Mat> &img,
                                   vector<Blob<float> *> &outputs,
                                   vector<vector<Detection> > &detect_results) {
    int tot_cnt = 0;
    int box_num = outputs[tot_cnt]->height();

    const float *top_data = outputs[tot_cnt]->cpu_data();

    int image_offset = detect_results.size();
    for (int i = 0; i < img.size(); ++i) {
        vector<Detection> imageDetection;
        detect_results.push_back(imageDetection);
    }

    for (int j = 0; j < box_num; j++) {
        int img_id = top_data[j * 7 + 0];
        if (img_id < 0 || img_id >= detect_results.size()) {
            LOG(ERROR) << "Image id invalid: " << img_id << endl;
            continue;
        }
        vector<Detection> &imageDetection = detect_results[image_offset + img_id];

        int cls = top_data[j * 7 + 1];
        float score = top_data[j * 7 + 2];
        float xmin = top_data[j * 7 + 3] * img[img_id].cols;
        float ymin = top_data[j * 7 + 4] * img[img_id].rows;
        float xmax = top_data[j * 7 + 5] * img[img_id].cols;
        float ymax = top_data[j * 7 + 6] * img[img_id].rows;

        if (score > threshold_) {

            Detection detection;
            detection.box = Rect(xmin, ymin, xmax - xmin, ymax - ymin);
            detection.id = cls;
            detection.confidence = score;
            imageDetection.push_back(detection);

#ifdef SHOW_VIS
            char char_score[100];
            sprintf(char_score, "%.3f", score);
            rectangle(img[img_id], Rect(xmin, ymin, xmax - xmin, ymax - ymin), color_[cls]);
            putText(img[img_id],
                    tags_[cls] + "_" + string(char_score),
                    Point(xmin, ymin),
                    CV_FONT_HERSHEY_COMPLEX,
                    0.5,
                    color_[0]);
#endif

        }
    }
#ifdef  SHOW_VIS
    for(int i = 0; i < img.size(); ++i){
        cv::Mat image = img[i];
        imshow("debug.jpg", image);
        waitKey(-1);
    }
#endif
}
int VehicleCaffeDetector::DetectBatch(vector<cv::Mat> &img,
                                      vector<vector<Detection> > &detect_results) {

    if (!device_setted_) {
        Caffe::set_mode(Caffe::GPU);
        Caffe::SetDevice(gpu_id_);
        device_setted_ = true;
    }


    detect_results.clear();
    vector<cv::Mat> toPredict;
    for (int i = 0; i < img.size(); ++i) {
        cv::Mat image = img[i];
        toPredict.push_back(image);
        if (toPredict.size() == batch_size_) {
            vector<Blob<float> *> outputs = PredictBatch(toPredict);
            Fullfil(toPredict, outputs, detect_results);
            toPredict.clear();
        }
    }

    if (toPredict.size() > 0) {
        vector<Blob<float> *> outputs = PredictBatch(toPredict);
        Fullfil(toPredict, outputs, detect_results);
    }

//    // make sure batch size is times of the batch size
//    if (img.size() % batch_size_ != 0) {
//        int batchShort = batch_size_ - (img.size() % batch_size_);
//        for (int i = 0; i < batchShort; ++i) {
//            DLOG(INFO) << "Input images size less than batch size" << endl;
//            img.push_back(cv::Mat(1, 1, CV_8UC3));
//        }
//    }
//
//    detect_results.clear();
//    vector<cv::Mat> toPredict;
//    for (int i = 0; i < img.size(); ++i) {
//        cv::Mat image = img[i];
//        toPredict.push_back(image);
//        if (toPredict.size() == batch_size_) {
//            vector<Blob<float> *> outputs = PredictBatch(toPredict);
//            Fullfil(toPredict, outputs, detect_results);
//            toPredict.clear();
//        }
//
//    }

}

std::vector<Blob<float> *> VehicleCaffeDetector::PredictBatch(const vector<Mat> &imgs) {

    vector<Blob<float> *> outputs;

    Blob<float> *input_layer = net_->input_blobs()[0];
    float *input_data = input_layer->mutable_cpu_data();

    if (imgs.size() <= batch_size_) {
        input_layer->Reshape(imgs.size(), num_channels_,
                             input_geometry_.height,
                             input_geometry_.width);
        net_->Reshape();
    } else {
        LOG(ERROR) << "Input images size is more than batch size!" << endl;
        return outputs;
    }
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