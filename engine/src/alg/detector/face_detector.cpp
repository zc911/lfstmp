#include "face_detector.h"
#include "alg/caffe_helper.h"
#include "face_detector.h"
namespace dg {

bool mycmp(Detection b1, Detection b2) {
    return b1.confidence > b2.confidence;
}

FaceDetector::FaceDetector(const FaceDetectorConfig &config)
    : scale_(config.scale),
      batch_size_(config.batch_size),
      conf_thres_(config.confidence) {
    use_gpu_ = config.use_gpu;
    if (use_gpu_) {
        Caffe::SetDevice(config.gpu_id);
        Caffe::set_mode(Caffe::GPU);
        use_gpu_ = true;
    } else {
        Caffe::set_mode(Caffe::CPU);
        use_gpu_ = false;
    }

    LOG(INFO) << "loading model file: " << config.deploy_file;
#if DEBUG
    net_.reset(
        new Net<float>(config.deploy_file, TEST));
#else
    net_.reset(
            new Net<float>(config.deploy_file, TEST, config.is_model_encrypt));
#endif
    LOG(INFO) << "loading trained file : " << config.model_file;
    net_->CopyTrainedLayersFrom(config.model_file);

    CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";

    Blob<float> *input_layer = net_->input_blobs()[0];

    num_channels_ = input_layer->channels();
    CHECK(num_channels_ == 3 || num_channels_ == 1)
    << "Input layer should have 1 or 3 channels.";
    pixel_means_.push_back(102.9801);
    pixel_means_.push_back(115.9465);
    pixel_means_.push_back(122.7717);

    layer_name_cls_ = "conv_face_16_cls";
    layer_name_reg_ = "conv_face_16_reg";
    sliding_window_stride_ = 16;
    area_.push_back(2 * 24 * 24);
    area_.push_back(48 * 48);
    area_.push_back(2 * 48 * 48);
    area_.push_back(96 * 96);
    area_.push_back(2 * 96 * 96);
    ratio_.push_back(1);

    vector<int> shape;
    shape.push_back(batch_size_);
    shape.push_back(3);
    shape.push_back(scale_);
    shape.push_back(scale_);
    input_layer->Reshape(shape);
    net_->Reshape();
 /*   const vector<boost::shared_ptr<Layer<float> > > &layers = net_->layers();
    const vector<vector<Blob<float> *> > &bottom_vecs = net_->bottom_vecs();
    const vector<vector<Blob<float> *> > &top_vecs = net_->top_vecs();
    for (int i = 0; i < layers.size(); ++i) {
        layers[i]->Forward(bottom_vecs[i], top_vecs[i]);
    }*/
}

FaceDetector::~FaceDetector() {

}

void FaceDetector::Forward(const vector<cv::Mat> &imgs,
                           vector<Blob<float> *> &outputs) {
    Blob<float> *input_layer = net_->input_blobs()[0];

    int max_width = 0;
    int max_height = 0;
    vector<Mat> samples;

    for (int i = 0; i < imgs.size(); ++i) {
        Mat img = imgs[i];
        Mat sample;
        float resize_ratio = ReScaleImage(img, scale_);
        CheckChannel(img, num_channels_, sample);
        resize_ratios_.push_back(resize_ratio);
        samples.push_back(sample);

        if (max_width < sample.cols) {
            max_width = sample.cols;
        }
        if (max_height < sample.rows) {
            max_height = sample.rows;
        }
    }

    vector<int> shape;
    shape.push_back(imgs.size());
    shape.push_back(3);
    shape.push_back(max_height);
    shape.push_back(max_width);
    input_layer->Reshape(shape);
    net_->Reshape();

    resize_ratios_.clear();

    for (size_t i = 0; i < samples.size(); ++i) {

//        if (resized.channels() == 3 && num_channels_ == 1)
//            cvtColor(resized, sample, CV_BGR2GRAY);
//        else if (resized.channels() == 4 && num_channels_ == 1)
//            cvtColor(resized, sample, CV_BGRA2GRAY);
//        else if (resized.channels() == 4 && num_channels_ == 3)
//            cvtColor(resized, sample, CV_RGBA2BGR);
//        else if (resized.channels() == 1 && num_channels_ == 3)
//            cvtColor(resized, sample, CV_GRAY2BGR);
//        else
//            sample = resized;

        Mat sample = samples[i];
        float *input_data = input_layer->mutable_cpu_data();
        size_t image_off = i * sample.channels() * sample.rows * sample.cols;
        for (int k = 0; k < sample.channels(); ++k) {
            size_t channel_off = k * sample.rows * sample.cols;
            for (int row = 0; row < sample.rows; ++row) {
                size_t row_off = row * sample.cols;
                for (int col = 0; col < sample.cols; ++col) {
                    input_data[image_off + channel_off + row_off + col] = float(
                        sample.at<uchar>(row, col * 3 + k))
                        - pixel_means_[k];
                }
            }
        }
    }

    net_->ForwardPrefilled();

    if (use_gpu_ == true) {
        cudaDeviceSynchronize();
    }

    outputs.resize(0);
    Blob<float> *output_cls = net_->blob_by_name(layer_name_cls_).get();
    Blob<float> *output_reg = net_->blob_by_name(layer_name_reg_).get();
    outputs.push_back(output_cls);
    outputs.push_back(output_reg);
}

void FaceDetector::NMS(vector<Detection> &p, float threshold) {
    sort(p.begin(), p.end(), mycmp);
    for (size_t i = 0; i < p.size(); ++i) {
        if (p[i].deleted)
            continue;
        for (size_t j = i + 1; j < p.size(); ++j) {

            if (!p[j].deleted) {
                cv::Rect intersect = p[i].box & p[j].box;
                float iou = intersect.area() * 1.0f / p[j].box.area();
                if (iou > threshold) {
                    p[j].deleted = true;
                }
            }
        }
    }
}

vector<vector<Detection>> FaceDetector::Detect(vector<Mat> imgs) {
    vector<Blob<float> *> outputs;
    Forward(imgs, outputs);
    vector<vector<Detection>> boxes;
    GetDetection(outputs, boxes);
    return boxes;
}

void FaceDetector::GetDetection(vector<Blob<float> *> &outputs,
                                vector<vector<Detection> > &final_vbbox) {
    Blob<float> *cls = outputs[0];
    Blob<float> *reg = outputs[1];
    final_vbbox.clear();
    final_vbbox.resize(0);
    final_vbbox.resize(cls->num());
    int scale_num = area_.size() * ratio_.size();

    assert(cls->channels() == scale_num * 2);
    assert(reg->channels() == scale_num * 4);

    assert(cls->height() == reg->height());
    assert(cls->width() == reg->width());

    vector<Detection> vbbox;
    const float *cls_cpu = cls->cpu_data();
    const float *reg_cpu = reg->cpu_data();

    vector<float> gt_ww, gt_hh;
    gt_ww.resize(scale_num);
    gt_hh.resize(scale_num);

    for (size_t i = 0; i < area_.size(); ++i) {
        for (size_t j = 0; j < ratio_.size(); ++j) {
            int index = i * ratio_.size() + j;
            gt_ww[index] = sqrt(area_[i] * ratio_[j]);
            gt_hh[index] = gt_ww[index] / ratio_[j];
        }
    }
    int cls_index = 0;
    int reg_index = 0;
    for (int img_idx = 0; img_idx < cls->num(); ++img_idx) {
        vbbox.clear();
        for (int scale_idx = 0; scale_idx < scale_num; ++scale_idx) {
            int skip = cls->height() * cls->width();
            for (int h = 0; h < cls->height(); ++h) {
                for (int w = 0; w < cls->width(); ++w) {
                    float confidence;
                    float rect[4] = {};
                    {
                        float x0 = cls_cpu[cls_index];
                        float x1 = cls_cpu[cls_index + skip];
                        float min_01 = min(x1, x0);
                        x0 -= min_01;
                        x1 -= min_01;
                        confidence = exp(x1) / (exp(x1) + exp(x0));
                    }
                    if (confidence > conf_thres_) {
                        for (int j = 0; j < 4; ++j) {
                            rect[j] = reg_cpu[reg_index + j * skip];
                        }

                        float shift_x = w * sliding_window_stride_
                            + sliding_window_stride_ / 2.f - 1;
                        float shift_y = h * sliding_window_stride_
                            + sliding_window_stride_ / 2.f - 1;
                        rect[2] = exp(rect[2]) * gt_ww[scale_idx];
                        rect[3] = exp(rect[3]) * gt_hh[scale_idx];
                        rect[0] = rect[0] * gt_ww[scale_idx] - rect[2] / 2.f
                            + shift_x;
                        rect[1] = rect[1] * gt_hh[scale_idx] - rect[3] / 2.f
                            + shift_y;

                        Detection bbox;
                        bbox.confidence = confidence;
                        bbox.box = Rect(rect[0], rect[1], rect[2], rect[3]);
                        //bbox.box &= Rect(0, 0, scale_, scale_);
                        bbox.deleted = false;
                        vbbox.push_back(bbox);
                    }

                    cls_index += 1;
                    reg_index += 1;
                }
            }
            cls_index += skip;
            reg_index += 3 * skip;
        }
        NMS(vbbox, 0.2);
        for (size_t i = 0; i < vbbox.size(); ++i) {
            Detection detection = vbbox[i];

            if (!detection.deleted) {
                float ratio = resize_ratios_[img_idx];
                detection.Rescale(ratio);
                final_vbbox[img_idx].push_back(detection);
            }
        }
    }
}

} /* namespace dg */
