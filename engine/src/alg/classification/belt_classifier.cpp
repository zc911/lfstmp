//
// Created by jiajaichen on 16-9-5.
//

#include "belt_classifier.h"
#include "alg/caffe_helper.h"

namespace dg {
CaffeBeltClassifier::CaffeBeltClassifier(const VehicleBeltConfig &config)
    : device_setted_(false),
      caffe_config_(config),
      rescale_(100) {

    device_setted_ = false;

    if (caffe_config_.use_gpu) {
        Caffe::SetDevice(caffe_config_.gpu_id);
        Caffe::set_mode(Caffe::GPU);
        LOG(INFO) << "Use device " << caffe_config_.gpu_id << endl;

    } else {
        LOG(WARNING) << "Use CPU only" << endl;
        Caffe::set_mode(Caffe::CPU);
    }

    string deploy_content;
    ModelsMap *modelsMap = ModelsMap::GetInstance();

    modelsMap->getModelContent(config.deploy_file, deploy_content);
    net_.reset(
        new Net<float>(config.deploy_file, deploy_content, TEST));
    string model_content;
    modelsMap->getModelContent(config.model_file, model_content);
    net_->CopyTrainedLayersFrom(config.model_file, model_content);

    CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";

    Blob<float> *input_layer = net_->input_blobs()[0];
    input_geometry_ = cv::Size(input_layer->width(), input_layer->height());
    num_channels_ = input_layer->channels();
    means_ = cv::Mat(input_geometry_, CV_32FC3, Scalar(128, 128, 128));
    input_layer->Reshape(caffe_config_.batch_size, num_channels_,
                         input_geometry_.height, input_geometry_.width);
    /* Forward dimension change to all layers. */
    net_->Reshape();
    /*   const vector<boost::shared_ptr<Layer<float> > > &layers = net_->layers();
       const vector<vector<Blob<float> *> > &bottom_vecs = net_->bottom_vecs();
       const vector<vector<Blob<float> *> > &top_vecs = net_->top_vecs();
       for (int i = 0; i < layers.size(); ++i) {
           layers[i]->Forward(bottom_vecs[i], top_vecs[i]);
       }*/
}
CaffeBeltClassifier::~CaffeBeltClassifier() {

}


vector<vector<Prediction> > CaffeBeltClassifier::ClassifyAutoBatch(const vector<Mat> &imgs) {

    vector<vector<Prediction> > prediction;
    vector<Mat> images;

    for (auto sample : imgs) {
        Mat flipped_img = flip_(sample);

        sample = adap_histeq(sample);
        Mat tmp = center_crop(sample, 16);
        images.push_back(tmp);
        tmp = random_crop(sample, 16);
        images.push_back(tmp);
        flipped_img = adap_histeq(flipped_img);
        tmp = center_crop(flipped_img, 16);
        images.push_back(tmp);
        tmp = random_crop(flipped_img, 16);
        images.push_back(tmp);
    }
    for (auto batch_images : PrepareBatch(images, caffe_config_.batch_size)) {

        vector<vector<Prediction> > pred = ClassifyBatch(batch_images);
        prediction.insert(prediction.end(), pred.begin(), pred.end());
    }
    int padding_size = (caffe_config_.batch_size
                        - imgs.size() % caffe_config_.batch_size)
                       % caffe_config_.batch_size;
    prediction.erase(prediction.end() - padding_size, prediction.end());
    return prediction;
}

vector<vector<Prediction> > CaffeBeltClassifier::ClassifyBatch(
    const vector<Mat> &imgs) {
    vector<Blob<float> *> output_layer = PredictBatch(imgs);
    int class_num_ = output_layer[0]->channels();
    const float *begin = output_layer[0]->cpu_data();
    const float *end = begin + output_layer[0]->channels() * imgs.size();
    vector<float> output_batch = std::vector<float>(begin, end);
    std::vector<std::vector<Prediction> > predictions;

    for (int j = 0; j < imgs.size(); j++) {
        std::vector<float> output(output_batch.begin() + j * class_num_,
                                  output_batch.begin() + (j + 1) * class_num_);
        std::vector<Prediction> prediction_single;


        for (int i = 0; i < class_num_; ++i) {
            prediction_single.push_back(std::make_pair(i, output[i]));
        }
        predictions.push_back(std::vector<Prediction>(prediction_single));
    }
    return predictions;
}

vector<Blob<float> *> CaffeBeltClassifier::PredictBatch(
    const vector<Mat> imgs) {

    if (!device_setted_) {
        Caffe::SetDevice(caffe_config_.gpu_id);
        Caffe::set_mode(Caffe::GPU);
        device_setted_ = true;
    }

    std::vector<std::vector<cv::Mat> > input_batch;
    Blob<float>* input_layer = net_->input_blobs()[0];
    input_layer->Reshape(caffe_config_.batch_size, num_channels_, input_geometry_.height,
                         input_geometry_.width);
    float* input_data = input_layer->mutable_cpu_data();
    Mat sample = img_list[j];
    if ((sample.rows != input_geometry_.height)
            || (sample.cols != input_geometry_.width)) {
        resize(sample, sample, Size(input_geometry_.width, input_geometry_.height));
    }
    int cnt = 0;
    for (int k = 0; k < sample.channels(); k++) {
        for (int r = 0; r < sample.rows; r++) {
            for (int c = 0; c < sample.cols; c++) {
                input_data[cnt++] = (float(sample.at<uchar>(r, c * 3 + k)) - means_[k]) * 0.01;
            }
        }
    }
    net_->ForwardPrefilled();
    if (caffe_config_.use_gpu) {
        cudaDeviceSynchronize();
    }

    /* Copy the output layer to a std::vector */
    vector<Blob<float> *> outputs;
    for (int i = 0; i < net_->num_outputs(); i++) {
        Blob<float> *output_layer = net_->output_blobs()[i];
        outputs.push_back(output_layer);
    }

    return outputs;

}

}