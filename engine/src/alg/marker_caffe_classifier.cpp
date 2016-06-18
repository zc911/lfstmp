/*
 * marker_caffe_classifier.cpp
 *
 *  Created on: Apr 21, 2016
 *      Author: jiajiachen
 */

#include "marker_caffe_classifier.h"

namespace dg {
MarkerCaffeClassifier::MarkerCaffeClassifier(MarkerConfig &markerconfig)
        : device_setted_(false),
          marker_config_(markerconfig),
          means_( { 128, 128, 128 }),
          rescale_(1) {

    if (marker_config_.use_gpu) {
        Caffe::SetDevice(marker_config_.gpu_id);
        Caffe::set_mode(Caffe::GPU);
        LOG(INFO)<< "Use device " << marker_config_.gpu_id << endl;

    } else {
        LOG(WARNING) << "Use CPU only" << endl;
        Caffe::set_mode(Caffe::CPU);
    }

    /* Load the network. */
    net_.reset(
            new Net<float>(markerconfig.deploy_file, TEST));
    net_->CopyTrainedLayersFrom(marker_config_.model_file);
    CHECK_EQ(net_->num_inputs(), 1)<< "Network should have exactly one input.";
    //   CHECK_EQ(net_->num_outputs(), 1)<< "Network should have exactly one output.";

    Blob<float>* input_layer = net_->input_blobs()[0];
    num_channels_ = input_layer->channels();
    CHECK(num_channels_ == 3 || num_channels_ == 1)
    << "Input layer should have 1 or 3 channels.";
    input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

    setupMarker();

}
void MarkerCaffeClassifier::setupMarker() {
    {
        float area[] = { 200, 400, 800, 1600 };
        float ratio[] = { 0.9, 1.2 };
        Marker marker;
        marker.id = MOT;
        marker.stride = 8;
        vector<float> areav(area, area + sizeof(area) / sizeof(float));
        marker.area = areav;
        vector<float> ratiov(ratio, ratio + sizeof(ratio) / sizeof(float));
        marker.ratio = ratiov;
        marker.threshold = 0.2;
        marker.max = 100;
        marker.confidence = marker_config_.marker_confidence[MOT];
        markers_.insert(pair<int, Marker>(marker.id, marker));

    }
    {
        float area[] = { 800, 1600, 3200 };
        float ratio[] = { 0.2, 0.35 };
        Marker marker;
        marker.id = Accessories;
        marker.stride = 8;
        vector<float> areav(area, area + sizeof(area) / sizeof(float));
        marker.area = areav;
        vector<float> ratiov(ratio, ratio + sizeof(ratio) / sizeof(float));
        marker.ratio = ratiov;
        marker.threshold = 0.01;
        marker.max = 1;
        marker.confidence = marker_config_.marker_confidence[Accessories];
        markers_.insert(pair<int, Marker>(marker.id, marker));
    }

    {
        float area[] = { 1600, 3200, 6400, 12800 };
        float ratio[] = { 0.5, 1, 1.5 };
        Marker marker;
        marker.id = TissueBox;
        marker.stride = 16;
        vector<float> areav(area, area + sizeof(area) / sizeof(float));
        marker.area = areav;
        vector<float> ratiov(ratio, ratio + sizeof(ratio) / sizeof(float));
        marker.ratio = ratiov;
        marker.threshold = 0.2;
        marker.max = 5;
        marker.confidence = marker_config_.marker_confidence[TissueBox];
        markers_.insert(pair<int, Marker>(marker.id, marker));
    }
    {
        float area[] = { 1600, 3200, 6400, 12800 };
        float ratio[] = { 0.5, 1, 1.5 };
        Marker marker;
        marker.id = Belt;
        marker.stride = 16;
        vector<float> areav(area, area + sizeof(area) / sizeof(float));
        marker.area = areav;
        vector<float> ratiov(ratio, ratio + sizeof(ratio) / sizeof(float));
        marker.ratio = ratiov;
        marker.threshold = 0.01;
        marker.max = 2;
        marker.confidence = marker_config_.marker_confidence[Belt];
        markers_.insert(pair<int, Marker>(marker.id, marker));
    }
    {
        float area[] = { 400, 800, 1600, 3200, 6400 };
        float ratio[] = { 0.5, 1, 2, 4 };
        Marker marker;
        marker.id = Others;
        marker.stride = 16;
        vector<float> areav(area, area + sizeof(area) / sizeof(float));
        marker.area = areav;
        vector<float> ratiov(ratio, ratio + sizeof(ratio) / sizeof(float));
        marker.ratio = ratiov;
        marker.threshold = 0.2;
        marker.max = 10;
        marker.confidence = marker_config_.marker_confidence[Others];
        markers_.insert(pair<int, Marker>(marker.id, marker));
    }

    {
        float area[] = { 800, 1600, 3200, 6400 };
        float ratio[] = { 3, 5 };
        Marker marker;
        marker.id = SunVisor;
        marker.stride = 16;
        vector<float> areav(area, area + sizeof(area) / sizeof(float));
        marker.area = areav;
        vector<float> ratiov(ratio, ratio + sizeof(ratio) / sizeof(float));
        marker.ratio = ratiov;
        marker.color = Scalar(0, 255, 0);
        marker.threshold = 0.01;
        marker.max = 2;
        marker.confidence = marker_config_.marker_confidence[SunVisor];
        markers_.insert(pair<int, Marker>(marker.id, marker));
    }
}
vector<vector<Detection> > MarkerCaffeClassifier::ClassifyBatch(
        vector<Mat> imgs) {
    int batch_size = imgs.size();
    vector<Mat> tiny_images;
    vector<float> enlarge_ratios;
    for (int i = 0; i < batch_size; i++) {

        Mat img = imgs[i];
        int max_size = max(img.rows, img.cols);
        int min_size = min(img.rows, img.cols);
        float enlarge_ratio = marker_config_.target_min_size / min_size;

        if (max_size * enlarge_ratio > marker_config_.target_max_size) {
            enlarge_ratio = marker_config_.target_max_size / max_size;
        }
        int target_row = img.rows * enlarge_ratio;
        int target_col = img.cols * enlarge_ratio;
        enlarge_ratios.push_back(enlarge_ratio);

        resize(img, img, Size(target_col, target_row));

        tiny_images.push_back(img);

    }

    vector<Blob<float>*> tiny_outputs = PredictBatch(tiny_images);

    vector<vector<Detection> > preds(batch_size);

    for (map<int, Marker>::iterator it = markers_.begin(); it != markers_.end();
            it++) {
        Blob<float> *cls = tiny_outputs[2 * (it->second).id];
        Blob<float> *reg = tiny_outputs[2 * (it->second).id + 1];
        vector<vector<Detection> > rpns = get_final_bbox(tiny_images, cls, reg,
                                                         enlarge_ratios,
                                                         it->second, imgs);
        for (int i = 0; i < batch_size; i++) {
            preds[i].insert(preds[i].end(), rpns[i].begin(), rpns[i].end());
        }
    }

    return preds;
}
vector<vector<Detection> > MarkerCaffeClassifier::ClassifyAutoBatch(
        vector<Mat> imgs) {
    vector<vector<Detection> > prediction;
    vector<Mat> images = imgs;
    for (auto batch_images : PrepareBatch(images, marker_config_.batch_size)) {
        vector<vector<Detection> > pred = ClassifyBatch(batch_images);
        prediction.insert(prediction.end(), pred.begin(), pred.end());
    }
    int padding_size = (marker_config_.batch_size
            - imgs.size() % marker_config_.batch_size)
            % marker_config_.batch_size;
    prediction.erase(prediction.end() - padding_size, prediction.end());
    return prediction;
}

vector<vector<Detection> > MarkerCaffeClassifier::get_final_bbox(
        vector<Mat> images, Blob<float>* cls, Blob<float>* reg,
        vector<float> enlarge_ratios, Marker &marker, vector<Mat> origin_imgs) {

    int scale_num = marker.area.size() * marker.ratio.size();

    cls->Reshape(cls->num() * scale_num, cls->channels() / scale_num,
                 cls->height(), cls->width());

    reg->Reshape(reg->num() * scale_num, reg->channels() / scale_num,
                 reg->height(), reg->width());
    int batch_size = images.size();

    vector<vector<Detection> > vvbbox(batch_size);
    const float* cls_cpu = cls->cpu_data();
    const float* reg_cpu = reg->cpu_data();

    float *gt_ww = new float[scale_num * batch_size];
    float *gt_hh = new float[scale_num * batch_size];

    int cnt = 0;
    for (int idx = 0; idx < images.size(); idx++) {

        for (int i = 0; i < marker.area.size(); i++) {
            for (int j = 0; j < marker.ratio.size(); j++) {
                gt_ww[cnt] = sqrt(marker.area[i] * marker.ratio[j]);
                //  * enlarge_ratios[idx];  //* global_ratio;
                gt_hh[cnt] = gt_ww[cnt] / marker.ratio[j];  //* enlarge_ratios[idx];  // * global_ratio;
                cnt++;
            }
        }
    }

    int cls_index = 0;
    int reg_index = 0;
    for (int i = 0; i < cls->num(); i++) {  // = batchsize * 25

        int skip = cls->height() * cls->width();
        for (int h = 0; h < cls->height(); h++) {
            for (int w = 0; w < cls->width(); w++) {
                float confidence;
                float rect[4] = { };
                float gt_cx = w * marker.stride;
                float gt_cy = h * marker.stride;
                {
                    float x0 = cls_cpu[cls_index];
                    float x1 = cls_cpu[cls_index + skip];
                    float min_01 = min(x1, x0);
                    x0 -= min_01;
                    x1 -= min_01;
                    confidence = exp(x1) / (exp(x1) + exp(x0));
                }
                if (confidence > marker_config_.global_confidence) {

                    for (int j = 0; j < 4; j++) {
                        rect[j] = reg_cpu[reg_index + j * skip];
                    }

                    float shift_x = 0, shift_y = 0;

                    shift_x = w * marker.stride + marker.stride / 2 - 1;
                    shift_y = h * marker.stride + marker.stride / 2 - 1;
                    rect[2] = exp(rect[2]) * gt_ww[i];
                    rect[3] = exp(rect[3]) * gt_hh[i];

                    rect[0] = rect[0] * gt_ww[i] - 0.5 * rect[2] + shift_x;
                    rect[1] = rect[1] * gt_hh[i] - 0.5 * rect[3] + shift_y;

                    Detection bbox;
                    Mat tmp = images[i * batch_size / cls->num()];
                    bbox.confidence = confidence;
                    bbox.box = Rect(rect[0], rect[1], rect[2], rect[3]);
                    bbox.box &= Rect(0, 0, tmp.cols, tmp.rows);
                    bbox.deleted = false;

                    vvbbox[i * batch_size / cls->num()].push_back(bbox);
                }

                cls_index += 1;
                reg_index += 1;
            }
        }
        cls_index += skip;
        reg_index += 3 * skip;

    }
    vector<vector<Detection> > final_vvbbox;

    for (int idx = 0; idx < vvbbox.size(); idx++) {
        sort(vvbbox[idx].begin(), vvbbox[idx].end(), detectionCmp);
        detectionNMS(vvbbox[idx], marker.threshold);
        vector<Detection> final_vbbox;
        for (int i = 0; i < vvbbox[idx].size(); i++) {

            if (!vvbbox[idx][i].deleted
                    && vvbbox[idx][i].confidence
                            > marker_config_.global_confidence) {
                //Rect box = vbbox[i].rect;
                Detection box = vvbbox[idx][i];

                float x = box.box.x / enlarge_ratios[idx];
                float y = box.box.y / enlarge_ratios[idx];
                float w = box.box.width / enlarge_ratios[idx];
                float h = box.box.height / enlarge_ratios[idx];

                box.box.x = x;
                box.box.y = y;
                box.box.width = w;
                box.box.height = h;
                box.id = marker.id;
                if (vvbbox[idx][i].confidence <= marker.confidence) {
                    continue;
                }

                if (final_vbbox.size() > marker.max)
                    break;
                if (filter(box, origin_imgs[idx].rows, origin_imgs[idx].cols))
                    final_vbbox.push_back(box);
            }

        }
        final_vvbbox.push_back(final_vbbox);
    }

    delete[] gt_ww;
    delete[] gt_hh;
    return final_vvbbox;
}
bool MarkerCaffeClassifier::filter(Detection box, int row, int col) {

    if (box.id == Accessories) {
        float ratio = box.box.x * 1.0 / col;
        if (ratio < marker_config_.accessories_x0
                || ratio > marker_config_.accessories_y0) {
            return false;
        }
    }

    if (box.id == MOT) {

        float ratiox = box.box.x * 1.0 / col;
        float ratioy = box.box.y * 1.0 / row;

        if (ratiox > marker_config_.mot_x1)
            return false;
        if (ratiox > marker_config_.mot_x0 && ratioy > marker_config_.mot_y0)
            return false;
        if (ratioy > marker_config_.mot_y1)
            return false;
    }
    if (box.id == SunVisor) {

        if (box.box.x / (float) col > marker_config_.sunVisor_x0
                && box.box.x / (float) col < marker_config_.sunVisor_x1) {
            return false;
        }
        if (box.box.y / (float) row > marker_config_.sunVisor_y0)
            return false;
    }
    if (box.id == Others) {
        if (box.box.y / (float) row < marker_config_.sunVisor_y1)
            return false;
    }
    return true;
}
vector<Blob<float>*> MarkerCaffeClassifier::PredictBatch(vector<Mat> imgs) {
    unsigned long long tt;

    if (!device_setted_) {
        Caffe::SetDevice(marker_config_.gpu_id);
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

    input_layer->Reshape(marker_config_.batch_size, num_channels_,
                         input_geometry_.height, input_geometry_.width);
    /* Forward dimension change to all layers. */
    net_->Reshape();

    float* input_data = input_layer->mutable_cpu_data();
    int cnt = 0;
    for (int i = 0; i < imgs.size(); i++) {
        cv::Mat sample;
        cv::Mat img = imgs[i];

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

        //if((sample.rows != input_geometry_.height) || (sample.cols != input_geometry_.width)) {
        //    cv::resize(sample, sample, Size(input_geometry_.width, input_geometry_.height));
        //}

        for (int k = 0; k < sample.channels(); k++) {
            for (int i = 0; i < sample.rows; i++) {
                for (int j = 0; j < sample.cols; j++) {
                    input_data[cnt] = (float(sample.at<uchar>(i, j * 3 + k))
                            - 128) / rescale_;
                    cnt += 1;
                }
            }
        }
    }

    net_->ForwardPrefilled();

    if (marker_config_.use_gpu) {
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

MarkerCaffeClassifier::~MarkerCaffeClassifier() {

}
}

