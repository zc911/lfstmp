/*
 * caffe_detector.cpp
 *
 *  Created on: Apr 18, 2016
 *      Author: jiajaichen
 */

#include "./car_only_caffe_detector.h"

namespace dg {

CarOnlyCaffeDetector::CarOnlyCaffeDetector(const VehicleCaffeDetectorConfig &config)

    : device_setted_(false), caffe_config_(config),
      scale_num_(21),
      rescale_(1) {

    if (config.use_gpu) {
        Caffe::SetDevice(config.gpu_id);
        Caffe::set_mode(Caffe::GPU);
    } else {
        Caffe::set_mode(Caffe::CPU);
    }
#if DEBUG
    net_.reset(
        new Net<float>(config.deploy_file, TEST));
#else
    net_.reset(
            new Net<float>(config.deploy_file, TEST, config.is_model_encrypt));
#endif
    net_->CopyTrainedLayersFrom(caffe_config_.model_file);


    Blob<float> *input_layer = net_->input_blobs()[0];
    num_channels_ = input_layer->channels();
    input_geometry_ = cv::Size(input_layer->width(), input_layer->height());
    means_[0] = 102.9801;
    means_[1] = 115.9265;
    means_[2] = 122.7717;
    net_->Reshape();
    /* const vector<boost::shared_ptr<Layer<float> > > &layers = net_->layers();
     const vector<vector<Blob<float> *> > &bottom_vecs = net_->bottom_vecs();
     const vector<vector<Blob<float> *> > &top_vecs = net_->top_vecs();
     for (int i = 0; i < layers.size(); ++i) {
         layers[i]->Forward(bottom_vecs[i], top_vecs[i]);
     }*/

}

CarOnlyCaffeDetector::~CarOnlyCaffeDetector() {
}

int CarOnlyCaffeDetector::DetectBatch(const vector<Mat> &batch,
                                      vector<vector<Detection> > &vvbbox) {
    if (!device_setted_ && caffe_config_.use_gpu) {
        device_setted_ = true;
        Caffe::SetDevice(caffe_config_.gpu_id);
        Caffe::set_mode(Caffe::GPU);
    }
    vector<Mat> toPredict;
    for (auto i : batch) {
        toPredict.push_back(i);
        if (toPredict.size() == caffe_config_.batch_size) {
            vector<vector<Detection>> result;
            DetectSolidBatch(toPredict, result);
            vvbbox.insert(vvbbox.end(), result.begin(), result.end());
            toPredict.clear();
        }
    }
    if (toPredict.size() > 0) {
        vector<vector<Detection>> result;
        DetectSolidBatch(toPredict, result);
        vvbbox.insert(vvbbox.end(), result.begin(), result.end());
    }


    return 1;

}
int CarOnlyCaffeDetector::DetectSolidBatch(const vector<Mat> &batch,
                                           vector<vector<Detection> > &vvbbox) {


    vector<Mat> images(batch);
    vector<Blob<float> *> outputs = PredictBatch(images);

    Blob<float> *cls = outputs[0];
    Blob<float> *reg = outputs[1];
    cls->Reshape(cls->num() * scale_num_, cls->channels() / scale_num_,
                 cls->height(), cls->width());

    reg->Reshape(reg->num() * scale_num_, reg->channels() / scale_num_,
                 reg->height(), reg->width());

    int batch_size = images.size();
    vvbbox.resize(batch_size);

    if (!(cls->num() == reg->num() && cls->num() == scale_num_ * batch_size)) {
        return 1;
    }

    if (cls->channels() != 2) {
        return 1;
    }
    if (reg->channels() != 4) {
        return 1;
    }

    if (cls->height() != reg->height()) {
        return 1;

    }
    if (cls->width() != reg->width()) {
        return 1;

    }
    cudaDeviceSynchronize();
    const float *cls_cpu = cls->cpu_data();
    const float *reg_cpu = reg->cpu_data();
    float mean[4] = {0, 0, 0, 0};
    float std[4] = {0.13848565, 0.13580033, 0.27823007, 0.26142551};

    float *gt_ww = new float[scale_num_ * images.size()];
    float *gt_hh = new float[scale_num_ * images.size()];
    float area[10] = {};
    for (int i = 0; i < scale_num_ / 3; i++) {
        area[i] = 50 * 50 * pow(2, i);
    }

    float ratio[3] = {0.5, 1.0, 2.0};  // w / h
    int cnt = 0;

    for (auto img : images) {

        int max_size = max(img.rows, img.cols);
        int min_size = min(img.rows, img.cols);

        float global_ratio = (float) min_size / (float) caffe_config_.target_min_size;

        if (global_ratio < (float) max_size / (float) caffe_config_.target_max_size) {
            global_ratio = (float) max_size / (float) caffe_config_.target_max_size;
        }
        for (int i = 0; i < scale_num_ / 3; i++) {
            for (int j = 0; j < 3; j++) {
                gt_ww[cnt] = sqrt(area[i] * ratio[j]);// * global_ratio;
                gt_hh[cnt] = gt_ww[cnt] / ratio[j];//* global_ratio;
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
                        confidence = exp(x1) / (exp(x1) + exp(x0));
                    }
                }

                float rect[4] = {};

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
                    Mat tmp = images[i * batch_size / cls->num()];
                    int max_size = max(tmp.rows, tmp.cols);
                    int min_size = min(tmp.rows, tmp.cols);

                    float global_ratio = min_size / caffe_config_.target_min_size;

                    if (global_ratio < max_size / caffe_config_.target_max_size) {
                        global_ratio = max_size / caffe_config_.target_max_size;
                    }
                    bbox.box = Rect((rect[0] - rect[2] / 2.0) * global_ratio,
                                    (rect[1] - rect[3] / 2.0) * global_ratio,
                                    rect[2] * global_ratio,
                                    rect[3] * global_ratio);


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

vector<Blob<float> *> CarOnlyCaffeDetector::PredictBatch(vector<Mat> imgs) {

    vector<Blob<float> *> outputs;
    if (imgs.size() > caffe_config_.batch_size) {
        return outputs;
    }

    Blob<float> *input_layer = net_->input_blobs()[0];
    int max_size = max(imgs[0].rows, imgs[0].cols);
    int min_size = min(imgs[0].rows, imgs[0].cols);

    float global_ratio = min_size / caffe_config_.target_min_size;

    if (global_ratio < max_size / caffe_config_.target_max_size) {
        global_ratio = max_size / caffe_config_.target_max_size;
    }
    input_geometry_.height = imgs[0].rows / global_ratio;  // + 100;
    input_geometry_.width = imgs[0].cols / global_ratio;  // + 100;

    for (int i = 0; i < imgs.size(); i++) {
        int max_size = max(imgs[i].rows, imgs[i].cols);
        int min_size = min(imgs[i].rows, imgs[i].cols);

        float global_ratio = min_size / caffe_config_.target_min_size;

        if (global_ratio < max_size / caffe_config_.target_max_size) {
            global_ratio = max_size / caffe_config_.target_max_size;
        }
        if (input_geometry_.height < imgs[i].rows / global_ratio) {
            input_geometry_.height = imgs[i].rows / global_ratio;
        }
        if (input_geometry_.width < imgs[i].cols / global_ratio) {
            input_geometry_.width = imgs[i].cols / global_ratio;
        }

    }


    input_layer->Reshape(imgs.size(), num_channels_, input_geometry_.height,
                         input_geometry_.width);
    net_->Reshape();

    float *input_data = input_layer->mutable_cpu_data();
    unsigned long long cnt = 0;
    for (auto img:imgs) {
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
        int max_size = max(sample.rows, sample.cols);
        int min_size = min(sample.rows, sample.cols);


        float global_ratio = (float) min_size / (float) caffe_config_.target_min_size;

        if (global_ratio < (float) max_size / (float) caffe_config_.target_max_size) {
            global_ratio = (float) max_size / (float) caffe_config_.target_max_size;
        }

        for (int k = 0; k < sample.channels(); k++) {
            for (int i = 0; i < (int) (sample.rows / global_ratio); i++) {
                for (int j = 0; j < (int) (sample.cols / global_ratio); j++) {
                    int indexi = (int) (i * global_ratio);
                    int indexj = (int) (j * global_ratio);
                    input_data[cnt + k * input_geometry_.width * input_geometry_.height + i * input_geometry_.width
                        + j] = (float(sample.at<uchar>(indexi, indexj * 3 + k))
                        - means_[k]);

                }
            }

        }
        cnt += input_geometry_.width * input_geometry_.height * 3;

    }
    net_->ForwardPrefilled();
    if (caffe_config_.use_gpu) {
        cudaDeviceSynchronize();
    }

    /* Copy the output layer to a std::vector */
    for (int i = 0; i < net_->num_outputs(); i++) {
        Blob<float> *output_layer = net_->output_blobs()[i];
        outputs.push_back(output_layer);
    }

    return outputs;

}

}
