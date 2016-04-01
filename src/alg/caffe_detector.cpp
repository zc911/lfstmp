/*
 * caffe_detector.cpp
 *
 *  Created on: Sep 28, 2015
 *      Author: irene
 */

#include "caffe_detector.h"

CaffeDetector::CaffeDetector(const string& model_file,
                             const string& trained_file, const bool use_GPU,
                             const int batch_size, const int gpuId)
        : Detector(model_file) {
    device_setted_ = false;
    if (use_GPU) {
        Caffe::SetDevice(gpuId);
        gpu_id_ = gpuId;
        Caffe::set_mode(Caffe::GPU);
        LOG(INFO)<< "Use device " << gpuId << endl;

    } else {
        LOG(WARNING) << "Use CPU only" << endl;
        Caffe::set_mode(Caffe::CPU);
    }

    /* Set batchsize */
    batch_size_ = batch_size;

    /* Load the network. */
    net_.reset(new Net<float>(model_file, TEST));
    net_->CopyTrainedLayersFrom(trained_file);

    CHECK_EQ(net_->num_inputs(), 1)<< "Network should have exactly one input.";
    //CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

    Blob<float>* input_layer = net_->input_blobs()[0];
    num_channels_ = input_layer->channels();
    CHECK(num_channels_ == 3 || num_channels_ == 1)
            << "Input layer should have 1 or 3 channels.";
    input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

    /* Load the binaryproto mean file. */
    //SetMean();
    rescale_ = 0;
    means_[0] = 102.9801;
    means_[1] = 115.9265;
    means_[2] = 122.7717;

    //TODO:can read from prototxt using cv method
    scale_num = 21;
    target_min_size_ = 500.0;
    target_max_size_ = 1000.0;
}

CaffeDetector::~CaffeDetector() {
}

static bool mycmp(BoundingBox b1, BoundingBox b2) {
    return b1.confidence > b2.confidence;
}

static void nms(vector<BoundingBox>& p, float threshold) {
    sort(p.begin(), p.end(), mycmp);
    int cnt = 0;
    for (int i = 0; i < p.size(); i++) {
        if (p[i].deleted)
            continue;
        cnt += 1;
        for (int j = i + 1; j < p.size(); j++) {
            if (!p[j].deleted) {
                cv::Rect intersect = p[i].rect & p[j].rect;
                float iou = intersect.area() * 1.0
                        / (p[i].rect.area() + p[j].rect.area()
                                - intersect.area());
                if (iou > threshold) {
                    p[j].deleted = true;
                }
            }
        }
    }
    //cout << "[after nms] left is " << cnt << endl;
}

vector<BoundingBox> CaffeDetector::Detect(const string image_filename,
                                          const int target_image_size) {
    vector<BoundingBox> a;
    return a;
}
vector<BoundingBox> CaffeDetector::Detect(const Mat & image,
                                          const int target_image_size) {
    vector<BoundingBox> vbbox;
    Mat img;
    img = image.clone();

    int max_size = max(img.rows, img.cols);
    int min_size = min(img.rows, img.cols);
    float enlarge_ratio = target_min_size_ / min_size;

    if (max_size * enlarge_ratio > target_max_size_) {
        enlarge_ratio = target_max_size_ / max_size;
    }

    int target_row = img.rows * enlarge_ratio;
    int target_col = img.cols * enlarge_ratio;

    DLOG(INFO)<< "[before]Image " << img.rows << " " << img.cols << endl;
    resize(img, img, Size(target_col, target_row), CV_INTER_LINEAR);
    vector<Mat> images;
    images.push_back(img);

    vector<Blob<float>*> outputs = PredictBatch(images);
    Blob<float>* cls = outputs[0];
    Blob<float>* reg = outputs[1];

    cls->Reshape(cls->num() * scale_num, cls->channels() / scale_num,
                 cls->height(), cls->width());
    reg->Reshape(reg->num() * scale_num, reg->channels() / scale_num,
                 reg->height(), reg->width());

    if (!(cls->num() == reg->num() && cls->num() == scale_num)) {
        return vbbox;
    }

    if (cls->channels() != 2) {
        return vbbox;
    }
    if (reg->channels() != 4) {
        return vbbox;
    }

    if (cls->height() != reg->height()) {
        return vbbox;

    }
    if (cls->width() != reg->width()) {
        return vbbox;

    }

    const float* cls_cpu = cls->cpu_data();
    const float* reg_cpu = reg->cpu_data();

    float mean[4] = { 0, 0, 0, 0 };
    float std[4] = { 0.13848565, 0.13580033, 0.27823007, 0.26142551 };

    float gt_ww[scale_num];
    float gt_hh[scale_num];
    float area[10] = { };
    for (int i = 0; i < scale_num / 3; i++) {
        area[i] = 50 * 50 * pow(2, i);
    }
    float ratio[3] = { 0.5, 1.0, 2.0 };  // w / h
    int cnt = 0;

    float global_ratio = 1.0 * min(images[0].rows, images[0].cols)
            / target_min_size_;
    for (int i = 0; i < scale_num / 3; i++) {
        for (int j = 0; j < 3; j++) {
            gt_ww[cnt] = sqrt(area[i] * ratio[j]) * global_ratio;
            gt_hh[cnt] = gt_ww[cnt] / ratio[j] * global_ratio;
            cnt++;
        }
    }
    for (int h = 0; h < cls->height(); h++) {
        for (int w = 0; w < cls->width(); w++) {
            for (int i = 0; i < cls->num(); i++) {

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

                rect[0] = rect[0] * gt_ww[i] + gt_cx;
                rect[1] = rect[1] * gt_hh[i] + gt_cy;
                rect[2] = exp(rect[2]) * gt_ww[i];
                rect[3] = exp(rect[3]) * gt_hh[i];

                if (confidence > 0.8) {
                    BoundingBox bbox;
                    bbox.confidence = confidence;
                    bbox.rect = Rect(rect[0] - rect[2] / 2.0,
                                     rect[1] - rect[3] / 2.0, rect[2], rect[3]);
                    bbox.rect &= Rect(0, 0, images[0].cols, images[0].rows);
                    bbox.deleted = false;
                    bbox.gt = Rect(gt_cx - gt_ww[i] / 2, gt_cy - gt_hh[i] / 2,
                                   gt_ww[i], gt_hh[i])
                            & Rect(0, 0, images[0].cols, images[0].rows);
                    vbbox.push_back(bbox);

                }
            }
        }
    }
    return vbbox;
}
void CaffeDetector::ChangeTargetSize(float target_min_size,
                                     float target_max_size) {
    target_min_size_ = target_min_size;
    target_max_size_ = target_max_size;
}
void CaffeDetector::SetMean() {

    mean_ = cv::Mat(input_geometry_, CV_32FC3,
                    Scalar(means_[0], means_[1], means_[2]));

    if (abs(means_[0] - 128) < 1 && abs(means_[1] - 128) < 1
            && abs(means_[2] - 128) < 1) {
        rescale_ = 1;
    }

}
void CaffeDetector::ChangeMean(float a, float b, float c) {
    means_[0] = a;
    means_[1] = b;
    means_[2] = c;
}
vector<Blob<float>*> CaffeDetector::PredictBatch(const vector<cv::Mat>& imgs) {
    if (!device_setted_) {
        Caffe::SetDevice(gpu_id_);
        device_setted_ = true;
    }

    Blob<float>* input_layer = net_->input_blobs()[0];
    input_geometry_.height = imgs[0].rows;
    input_geometry_.width = imgs[0].cols;
    input_layer->Reshape(batch_size_, num_channels_, input_geometry_.height,
                         input_geometry_.width);

    /* Forward dimension change to all layers. */
    SetMean();

    net_->Reshape();

    std::vector<std::vector<cv::Mat> > input_batch;
    WrapBatchInputLayer(&input_batch);
    PreprocessBatch(imgs, &input_batch);
    net_->ForwardPrefilled();
    /* Copy the output layer to a std::vector */
    vector<Blob<float>*> outputs;
    for (int i = 0; i < net_->num_outputs(); i++) {
        Blob<float>* output_layer = net_->output_blobs()[i];
        outputs.push_back(output_layer);
    }
    return outputs;
}

void CaffeDetector::WrapBatchInputLayer(
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

void CaffeDetector::PreprocessBatch(
        const vector<cv::Mat> imgs,
        std::vector<std::vector<cv::Mat> >* input_batch) {

    //SetMean();
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
            cv::resize(sample, sample_resized, input_geometry_,
                       CV_INTER_LINEAR);
        else
            sample_resized = sample;

        cv::Mat sample_float;
        if (num_channels_ == 3)
            sample_resized.convertTo(sample_float, CV_32FC3);
        else
            sample_resized.convertTo(sample_float, CV_32FC1);

        cv::Mat sample_normalized;
        cv::subtract(sample_float, mean_, sample_normalized);

        //cv::Mat sample_rescaled;
        if (rescale_) {
            cv::addWeighted(sample_normalized, 0.01, sample_normalized, 0, 0,
                            sample_normalized);
        }

        /* This operation will write the separate BGR planes directly to the
         * input layer of the network because it is wrapped by the cv::Mat
         * objects in input_channels. */
        //cv::split(sample_normalized, *input_channels);
        //cv::split(sample_rescaled, *input_channels);
        cv::split(sample_normalized, *input_channels);

//        CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
//              == net_->input_blobs()[0]->cpu_data())
//          << "Input channels are not wrapping the input layer of the network.";
    }
}
CaffeDetectorAdvance::CaffeDetectorAdvance(const string& model_file,
                                           const string& trained_file,
                                           const bool use_GPU,
                                           const int batch_size,
                                           const int gpuId)
        : CaffeDetector(model_file, trained_file, use_GPU, batch_size, gpuId) {
    ChangeMean(128, 128, 128);
}
vector<BoundingBox> CaffeDetectorAdvance::Detect(const string image_filename,
                                                 const int target_image_size) {
    vector<BoundingBox> a;
    return a;
}
vector<BoundingBox> CaffeDetectorAdvance::Detect(const Mat & image,
                                                 const int target_image_size) {
    vector<BoundingBox> result;

    sort(vbbox_.begin(), vbbox_.end(), mycmp);

    nms(vbbox_, 0.2);
    int tot_left = 0;
    //Mat resized_image = images[0].clone();
    Mat resized_image1 = image.clone();

    Mat resized_image = image.clone();
    int max_size = max(resized_image.rows, resized_image.cols);
    int min_size = min(resized_image.rows, resized_image.cols);
    float enlarge_ratio = target_min_size_ / min_size;

    if (max_size * enlarge_ratio > target_max_size_) {
        enlarge_ratio = target_max_size_ / max_size;
    }
    for (int i = 0; i < vbbox_.size(); i++) {
        if (!vbbox_[i].deleted && vbbox_[i].confidence > 0.8) {

            //              float cx = vbbox[i].gt.x + vbbox[i].gt.width / 2.0;
            //              float cy = vbbox[i].gt.y + vbbox[i].gt.height / 2.0;
            //
            //              if (cx < border_ratio/(1.+2*border_ratio)*resized_image.cols || cx > (1.+border_ratio)/(1.+2*border_ratio)*resized_image.cols) continue;
            //              if (cy < border_ratio/(1.+2*border_ratio)*resized_image.rows || cy > (1.+border_ratio)/(1.+2*border_ratio)*resized_image.rows) continue;
            //
            Rect box = vbbox_[i].rect;

            float x = box.x / enlarge_ratio, y = box.y / enlarge_ratio, w = box
                    .width / enlarge_ratio, h = box.height / enlarge_ratio;
            x -= w * 0.1;
            y -= h * 0.1;
            w *= 1.2;
            h *= 1.2;

            Rect newbox(x, y, w, h);
            //Mat img = resized_image(newbox & Rect(0,0, resized_image.cols, resized_image.rows));
            Mat img = image(newbox & Rect(0, 0, image.cols, image.rows));

            resize(img, img, Size(128, 128), CV_INTER_LINEAR);

            img = img(Rect(5, 5, 118, 118));
            vector<Mat> images;
            images.push_back(img);

            vector<Blob<float>*> outputs = PredictBatch(images);

            Blob<float>* cls = outputs[0];
            const float* cls_cpu = cls->cpu_data();
            if (cls_cpu[0] > 0.5 && vbbox_[i].confidence > 0.5) {

                //	rectangle(resized_image1, vbbox_[i].rect, Scalar(0, 0, 255)); // red
                tot_left++;

                //fprintf(fid, "%d %d %d %d %f ", vbbox[i].rect.x, vbbox[i].rect.y, vbbox[i].rect.width, vbbox[i].rect.height, vbbox[i].confidence);

            }
            if ((vbbox_[i].confidence > 0.995 && cls_cpu[0] > 0.9)
                    || vbbox_[i].confidence > 0.999) {
                x = box.x / enlarge_ratio, y = box.y / enlarge_ratio, w = box
                        .width / enlarge_ratio, h = box.height / enlarge_ratio;
                BoundingBox box;
//				box.border = NOBorderType;
                if (x < 0) {
//                    box.border = LeftType;
                }
                if ((w + x) > image.cols) {
//                    box.border = RightType;
                }
                x = x > 0 ? x : 0;
                y = y > 0 ? y : 0;
                w = (w + x) > image.cols ? (image.cols - x) : w;
                h = (h + y) > image.rows ? (image.rows - y) : h;
                box.rect = Rect(x, y, w, h);
                box.confidence = vbbox_[i].confidence;

                result.push_back(box);
                //		rectangle(resized_image1, box.rect, Scalar(255, 0, 0)); // red

            }

        }

    }
    //imshow("debug.jpg", resized_image1);
    //waitKey(-1);
    return result;
}

CaffeDetectorAdvance::~CaffeDetectorAdvance() {
}
void CaffeDetectorAdvance::setPrimaryResult(vector<BoundingBox>& vbbox) {
    vbbox_ = vbbox;

}
