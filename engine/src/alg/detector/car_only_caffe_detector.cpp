/*
 * caffe_detector.cpp
 *
 *  Created on: Apr 18, 2016
 *      Author: jiajaichen
 */

#include "./car_only_caffe_detector.h"

namespace dg {

CarOnlyCaffeDetector::CarOnlyCaffeDetector(const VehicleCaffeDetectorConfig &config)
          : device_setted_(false),
//            caffe_config_(config),
            scale_num_(21),
            means_(input_geometry_, CV_32FC3,
                   Scalar(102.9801, 115.9265, 122.7717)),
            rescale_(1) {

     if (config.use_gpu) {
          Caffe::SetDevice(config.gpu_id);
          Caffe::set_mode(Caffe::GPU);
          LOG(INFO)<< "Use device " << config.gpu_id << endl;

     } else {
          LOG(WARNING)<< "Use CPU only" << endl;
          Caffe::set_mode(Caffe::CPU);
     }

     /* Load the network. */
     net_.reset(new Net<float>(config.deploy_file, TEST));
     net_->CopyTrainedLayersFrom(config.model_file);

     CHECK_EQ(net_->num_inputs(), 1)<< "Network should have exactly one input.";
     //CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

     Blob<float>* input_layer = net_->input_blobs()[0];
     num_channels_ = input_layer->channels();
     CHECK(
               num_channels_ == 3 || num_channels_ == 1) << "Input layer should have 1 or 3 channels.";
     input_geometry_ = cv::Size(input_layer->width(), input_layer->height());
     /* Load the binaryproto mean file. */

}

CarOnlyCaffeDetector::~VehicleCaffeDetector() {
}

int CarOnlyCaffeDetector::DetectBatch(const vector<Mat> & batch,
                                      vector<vector<Detection> > &vvbbox) {
     if (device_setted_ && caffe_config_.use_gpu) {
          device_setted_ = false;
          Caffe::SetDevice(caffe_config_.gpu_id);
     }
     if (batch.size() != vvbbox.size()) {
          LOG(WARNING)<<"input size is not equal to output size"<<endl;
          return -1;
     }
     vector<Mat> images(batch);
     Mat img;
     int max_rows = 0, max_cols = 0;
     for (auto& iter_img : images) {
          img = iter_img;

          int max_size = max(img.rows, img.cols);
          int min_size = min(img.rows, img.cols);

          float enlarge_ratio = caffe_config_.target_min_size / min_size;

          if (max_size * enlarge_ratio > caffe_config_.target_max_size) {
               enlarge_ratio = caffe_config_.target_max_size / max_size;
          }

          int target_row = img.rows * enlarge_ratio;
          int target_col = img.cols * enlarge_ratio;

          resize(iter_img, iter_img, Size(target_col, target_row));
     }

     vector<Blob<float>*> outputs = PredictBatch(images);

     Blob<float>* cls = outputs[0];
     Blob<float>* reg = outputs[1];
     cls->Reshape(cls->num() * scale_num_, cls->channels() / scale_num_,
                  cls->height(), cls->width());

     reg->Reshape(reg->num() * scale_num_, reg->channels() / scale_num_,
                  reg->height(), reg->width());

     int batch_size = images.size();
     if (!(cls->num() == reg->num() && cls->num() == scale_num_ * batch_size)) {
          return vvbbox;
     }

     if (cls->channels() != 2) {
          return vvbbox;
     }
     if (reg->channels() != 4) {
          return vvbbox;
     }

     if (cls->height() != reg->height()) {
          return vvbbox;

     }
     if (cls->width() != reg->width()) {
          return vvbbox;

     }
     cudaDeviceSynchronize();
     const float* cls_cpu = cls->cpu_data();
     const float* reg_cpu = reg->cpu_data();
     float mean[4] = { 0, 0, 0, 0 };
     float std[4] = { 0.13848565, 0.13580033, 0.27823007, 0.26142551 };

     float* gt_ww = new float[scale_num_ * images.size()];
     float* gt_hh = new float[scale_num_ * images.size()];
     float area[10] = { };
     for (int i = 0; i < scale_num_ / 3; i++) {
          area[i] = 50 * 50 * pow(2, i);
     }

     float ratio[3] = { 0.5, 1.0, 2.0 };  // w / h
     int cnt = 0;

     for (int idx = 0; idx < images.size(); idx++) {
          float global_ratio = 1.0 * min(images[idx].rows, images[idx].cols)
                    / caffe_config_.target_min_size;
          for (int i = 0; i < scale_num_ / 3; i++) {
               for (int j = 0; j < 3; j++) {
                    gt_ww[cnt] = sqrt(area[i] * ratio[j]) * global_ratio;
                    gt_hh[cnt] = gt_ww[cnt] / ratio[j] * global_ratio;
                    cnt++;
               }
          }
     }
     for (int i = 0; i < cls->num(); i++) {
          for (int h = 0; h < cls->height(); h++) {
               for (int w = 0; w < cls->width(); w++) {

                    float confidence = 0;
                    for (int j = 0; j < cls->channels(); j++) {
                         int cls_index = i;
                         cls_index *= cls->channels();
                         cls_index += j;
                         cls_index *= cls->height();
                         cls_index += h;
                         cls_index *= cls->width();
                         cls_index += w;
                         if (j == 1) {
                              float x1 = cls_cpu[cls_index];
                              float x0 = cls_cpu[cls_index
                                        - cls->height() * cls->width()];
                              //x1 -= min(x1, x0);
                              //x0 -= min(x1, x0);
                              confidence = exp(x1) / (exp(x1) + exp(x0));
                         }
                    }

                    float rect[4] = { };

                    float gt_cx = w * 16.0;
                    float gt_cy = h * 16.0;

                    for (int j = 0; j < 4; j++) {
                         int reg_index = i;
                         reg_index *= reg->channels();
                         reg_index += j;
                         reg_index *= reg->height();
                         reg_index += h;
                         reg_index *= reg->width();
                         reg_index += w;

                         rect[j] = reg_cpu[reg_index] * std[j] + mean[j];
                    }

                    int gt_idx = i;  // % (cls->num() / batch_size);
                    rect[0] = rect[0] * gt_ww[gt_idx] + gt_cx;
                    rect[1] = rect[1] * gt_hh[gt_idx] + gt_cy;
                    rect[2] = exp(rect[2]) * gt_ww[gt_idx];
                    rect[3] = exp(rect[3]) * gt_hh[gt_idx];

                    if (confidence > 0.8) {
                         Detection bbox;
                         bbox.confidence = confidence;
                         bbox.box = Rect(rect[0] - rect[2] / 2.0,
                                         rect[1] - rect[3] / 2.0, rect[2],
                                         rect[3]);
                         Mat tmp = images[i * batch_size / cls->num()];
                         bbox.box &= Rect(0, 0, tmp.cols, tmp.rows);
                         bbox.deleted = false;
                         if (bbox.box.width == 0 || bbox.box.height == 0) {
                              continue;
                         }

                         vvbbox[i * batch_size / cls->num()].push_back(bbox);
                    }
               }
          }
     }

     delete[] gt_ww;
     delete[] gt_hh;

     return 1;
}

vector<Blob<float>*> CarOnlyCaffeDetector::PredictBatch(vector<Mat> imgs) {
     if (!device_setted_) {
          Caffe::SetDevice (gpu_id_);
          device_setted_ = true;
     }

     Blob<float>* input_layer = net_->input_blobs()[0];

     input_geometry_.height = imgs[0].rows;  // + 100;
     input_geometry_.width = imgs[0].cols;  // + 100;

     for (int i = 1; i < imgs.size(); i++) {
          if (input_geometry_.height < imgs[i].rows) {
               input_geometry_.height = imgs[i].rows;
          }
          if (input_geometry_.width < imgs[i].cols) {
               input_geometry_.width = imgs[i].cols;
          }

     }
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
     cudaDeviceSynchronize();

     return outputs;

}

void CarOnlyCaffeDetector::WrapBatchInputLayer(
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

void CarOnlyCaffeDetector::PreprocessBatch(
          const vector<cv::Mat> imgs,
          std::vector<std::vector<cv::Mat> >* input_batch) {

}
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
          cv::resize(sample, sample_resized, input_geometry_);
          cv::addWeighted(sample_resized, 0, sample_resized, 0, 0,
                          sample_resized);
          if (sample.size() != input_geometry_) {
               cv::Mat roi = sample_resized(
                         cv::Rect(0, 0, sample.cols, sample.rows));
               cv::addWeighted(roi, 0, sample, 1, 0, roi);
               //cv::resize(sample, sample_resized, input_geometry_);
          } else
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

          cv::split(sample_normalized, *input_channels);

     }
}
