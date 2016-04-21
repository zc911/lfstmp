/*
 * window_detector.cpp
 *
 *  Created on: Apr 19, 2016
 *      Author: jiajaichen
 */

#include "window_caffe_detector.h"
namespace dg {

WindowCaffeDetector::WindowCaffeDetector(CaffeConfig &config)
          : device_setted_(false),
            caffe_config_(caffe_config_),
            means_(input_geometry_, CV_32FC3, Scalar(128, 128, 128)),
            rescale_(100){

     if (config.use_gpu) {
          Caffe::SetDevice(config.gpu_id);
          Caffe::set_mode(Caffe::GPU);
          LOG(INFO)<< "Use device " << config.gpu_id << endl;

     } else {
          LOG(WARNING) << "Use CPU only" << endl;
          Caffe::set_mode(Caffe::CPU);
     }
     /* Set batchsize */

     /* Load the network. */
     net_.reset(new Net<float>(config.deploy_file, TEST));
     net_->CopyTrainedLayersFrom(config.model_file);
     CHECK_EQ(net_->num_inputs(), 1)<< "Network should have exactly one input.";
     //   CHECK_EQ(net_->num_outputs(), 1)<< "Network should have exactly one output.";

     Blob<float>* input_layer = net_->input_blobs()[0];
     num_channels_ = input_layer->channels();
     CHECK(num_channels_ == 3 || num_channels_ == 1)
     << "Input layer should have 1 or 3 channels.";
     input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

}
int WindowCaffeDetector::DetectBatch(
          const vector<cv::Mat> &imgs, const vector<cv::Mat> &resized_imgs,
          vector<vector<Detection> > &detect_results) {
     vector<Rect> crops;
     int padding_size = (caffe_config_.batch_size
               - imgs.size() % caffe_config_.batch_size)
               % caffe_config_.batch_size;
     auto images = PrepareBatch(imgs, caffe_config_.batch_size);
     auto resized_images = PrepareBatch(resized_imgs, caffe_config_.batch_size);

     unsigned long long tt;

     for (int i = 0; i < images.size(); i++) {
          vector<Rect> pred = Detect(resized_images[i], images[i]);
          crops.insert(crops.end(), pred.begin(), pred.end());
     }

     crops.erase(crops.end() - padding_size, crops.end());
     int cnt = 0;
     for (int i = 0; i < detect_results.size(); i++) {
          for (int j = 0; j < detect_results[i].size(); j++) {
               detect_results[i][j] = crops[cnt];
          }
     }
     return crops;

}

vector<Rect> WindowCaffeDetector::Detect(vector<Mat> resized_imgs,
                                         vector<Mat> imgs) {
     vector<Blob<float>*> window_outputs = PredictBatch(resized_imgs);

     Blob<float>* window_reg = window_outputs[0];
     const float* begin = window_reg->cpu_data();
     const float* end = begin
               + window_outputs[0]->channels() * resized_imgs.size();

     vector<float> output_batch = std::vector<float>(begin, end);
     vector<Rect> crops;
     vector<int> id;
     id.push_back(2);
     id.push_back(3);
     id.push_back(4);
     id.push_back(5);
     for (int k = 0; k < resized_imgs.size(); k++) {

          float lx = 100000, ly = 100000, rx = 0, ry = 0;

          float w_ratio = imgs[k].cols / 256.0;
          float h_ratio = imgs[k].rows / 256.0;
          for (int i = 0; i < id.size(); i++) {
               float x = output_batch[id[i] * 2
                         + k * window_outputs[0]->channels()];
               float y = output_batch[id[i] * 2 + 1
                         + k * window_outputs[0]->channels()];
               x *= 240;
               y *= 240;
               x += 8;
               y += 8;
               x *= w_ratio;
               y *= h_ratio;
               lx = min(x, lx);
               ly = min(y, ly);
               rx = max(x, rx);
               ry = max(y, ry);
          }
          int pad = int(min(imgs[k].rows, imgs[k].cols) * 0.1);
          lx -= pad;
          ly -= pad;
          rx += pad;
          ry += pad;
          Rect crop = Rect(lx, ly, rx - lx, ry - ly)
                    & Rect(0, 0, imgs[k].cols, imgs[k].rows);
          crops.push_back(crop);
     }

     return crops;
}
void WindowCaffeDetector::PreprocessBatch(
          const vector<cv::Mat> imgs,
          std::vector<std::vector<cv::Mat> >* input_batch) {
     for (int i = 0; i < imgs.size(); i++) {
          cv::Mat img = imgs[i];
          std::vector<cv::Mat> *input_channels = &(input_batch->at(i));

          /* Convert the input image to the input image format of the network. */
          cv::Mat sample;
          if (img.channels() == 3 && num_channels_ == 1)
               cv::cvtColor(img, sample, CV_BGR2GRAY);
          else if (img.channels() == 4 && num_channels_ == 1)
               cv::cvtColor(img, sample, CV_BGRA2GRAY);
          else if (img.channels() == 4 && num_channels_ == 3)
               cv::cvtColor(img, sample, CV_BGRA2BGR);
          else if (img.channels() == 1 && num_channels_ == 3)
               cv::cvtColor(img, sample, CV_GRAY2BGR);
          else
               sample = img;

          cv::Mat sample_resized;
          if (sample.size() != input_geometry_)
               cv::resize(sample, sample_resized, input_geometry_);
          else
               sample_resized = sample;

          cv::Mat sample_float;
          if (num_channels_ == 3)
               sample_resized.convertTo(sample_float, CV_32FC3);
          else
               sample_resized.convertTo(sample_float, CV_32FC1);

          cv::Mat sample_normalized;

          cv::subtract(sample_float, means_, sample_normalized);

          /* This operation will write the separate BGR planes directly to the
           * input layer of the network because it is wrapped by the cv::Mat
           * objects in input_channels. */
          if(rescale_==100)
          cv::addWeighted(sample_normalized, 0.01, sample_normalized, 0, 0,
                          sample_normalized);

          cv::split(sample_normalized, *input_channels);

     }
}
vector<Blob<float>*> WindowCaffeDetector::PredictBatch(const vector<Mat> imgs) {
     if (!device_setted_) {
          Caffe::SetDevice (gpu_id_);
          device_setted_ = true;
     }

     Blob<float>* input_layer = net_->input_blobs()[0];

     input_layer->Reshape(caffe_config_.batch_size, num_channels_,
                          input_geometry_.height, input_geometry_.width);
     /* Forward dimension change to all layers. */
     net_->Reshape();
     std::vector<std::vector<cv::Mat> > input_batch;
     WrapBatchInputLayer(&input_batch);
     PreprocessBatch(imgs, &input_batch);

     net_->ForwardPrefilled();

     if (caffe_config_.use_gpu) {
          cudaDeviceSynchronize();
     }

     /* Copy the output layer to a std::vector */
     vector<Blob<float>*> outputs;
     for (int i = 0; i < net_->num_outputs(); i++) {
          Blob<float>* output_layer = net_->output_blobs()[i];
          outputs.push_back(output_layer);
     }

     return outputs;

}
void WindowCaffeDetector::WrapBatchInputLayer(
        std::vector<std::vector<cv::Mat> > *input_batch) {
    Blob<float>* input_layer = net_->input_blobs()[0];

    int width = input_layer->width();
    int height = input_layer->height();
    int num = input_layer->num();
    float* input_data = input_layer->mutable_cpu_data();
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

WindowCaffeDetector::~WindowCaffeDetector() {
     // TODO Auto-generated destructor stub
}

}
