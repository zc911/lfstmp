//
// Created by jiajaichen on 16-6-24.
//

#include "caffe_vehicle_color.h"


namespace dg {
CaffeVehicleColor::CaffeVehicleColor(const CaffeVehicleColorConfig &config)
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

    net_.reset(
        new Net<float>(caffe_config_.deploy_file, TEST,
                       config.is_model_encrypt));

    //   net_.reset(
    //      new Net<float>(caffe_config_.deploy_file, TEST));

    net_->CopyTrainedLayersFrom(caffe_config_.model_file);
    CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";

    Blob<float> *input_layer = net_->input_blobs()[0];
    input_geometry_ = cv::Size(input_layer->width(), input_layer->height());
    num_channels_ = input_layer->channels();
    input_layer->Reshape(caffe_config_.batch_size, num_channels_,
                         input_geometry_.height, input_geometry_.width);
    /* Forward dimension change to all layers. */
    net_->Reshape();
    const vector<boost::shared_ptr<Layer<float> > >& layers = net_->layers();
    const vector<vector<Blob<float>*> >& bottom_vecs = net_->bottom_vecs();
    const vector<vector<Blob<float>*> >& top_vecs = net_->top_vecs();
    for(int i = 0; i < layers.size(); ++i) {
        layers[i]->Forward(bottom_vecs[i], top_vecs[i]);
    }
}

CaffeVehicleColor::~CaffeVehicleColor() {
}

vector<vector<Prediction> > CaffeVehicleColor::ClassifyAutoBatch(const vector<Mat> &imgs) {

    vector<vector<Prediction> > prediction;
    vector<Mat> images = imgs;
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

vector<vector<Prediction> > CaffeVehicleColor::ClassifyBatch(
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

vector<Blob<float> *> CaffeVehicleColor::PredictBatch(
    const vector<Mat> imgs) {

    if (!device_setted_) {
        Caffe::SetDevice(caffe_config_.gpu_id);
        device_setted_ = true;
    }

    std::vector<std::vector<cv::Mat> > input_batch;
    WrapBatchInputLayer(&input_batch);
    PreprocessBatch(imgs, &input_batch);
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

void CaffeVehicleColor::WrapBatchInputLayer(
    std::vector<std::vector<cv::Mat> > *input_batch) {
    Blob<float> *input_layer = net_->input_blobs()[0];

    int width = input_layer->width();
    int height = input_layer->height();
    int num = input_layer->num();
    float *input_data = input_layer->mutable_cpu_data();
    for (int j = 0; j < num; j++) {
        vector<cv::Mat> input_channels;
        for (int i = 0; i < input_layer->channels(); ++i) {
            cv::Mat channel(height, width, CV_32FC1, input_data);
            input_channels.push_back(channel);
            input_data += width * height;
        }
        input_batch->push_back(vector<cv::Mat>(input_channels));
    }

}

void CaffeVehicleColor::PreprocessBatch(
    const vector<cv::Mat> imgs,
    std::vector<std::vector<cv::Mat> > *input_batch) {
    for (int i = 0; i < imgs.size(); i++) {
        cv::Mat img = imgs[i];
        std::vector<cv::Mat> *input_channels = &(input_batch->at(i));
        cv::Mat result;
        dg::normalizeImage(img,result);
        /* Convert the input image to the input image format of the network. */
        cv::split(result, *input_channels);
    }
}

} /* namespace dg */