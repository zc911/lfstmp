#include "face_detector.h"
#include "alg/caffe_helper.h"
namespace dg {

bool mycmp(Detection b1, Detection b2) {
    return b1.confidence > b2.confidence;
}

FaceCaffeDetector::FaceCaffeDetector(const FaceDetectorConfig &config)
    : layer_name_cls_("conv_face_16_cls"),
      layer_name_reg_("conv_face_16_reg"),
      scale_(config.scale),
      img_scale_max_(config.img_scale_max),
      img_scale_min_(config.img_scale_min),
      batch_size_(config.batch_size),
      conf_thres_(config.confidence) {
    use_gpu_ = config.use_gpu;
    gpu_id_ = config.gpu_id;
    if (use_gpu_) {
        Caffe::SetDevice(config.gpu_id);
        Caffe::set_mode(Caffe::GPU);
        use_gpu_ = true;
    } else {
        Caffe::set_mode(Caffe::CPU);
        use_gpu_ = false;
    }

    ModelsMap *modelsMap = ModelsMap::GetInstance();
    string deploy_content;
    modelsMap->getModelContent(config.deploy_file, deploy_content);
    net_.reset(
        new Net<float>(config.deploy_file, deploy_content, TEST));
    string model_content;
    modelsMap->getModelContent(config.model_file, model_content);
    net_->CopyTrainedLayersFrom(config.model_file, model_content);


    CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";

    Blob<float> *input_layer = net_->input_blobs()[0];

    num_channels_ = input_layer->channels();
    CHECK(num_channels_ == 3 || num_channels_ == 1)
    << "Input layer should have 1 or 3 channels.";
    pixel_means_.push_back(128);
    pixel_means_.push_back(128);
    pixel_means_.push_back(128);

    area_.push_back(1 * 24 * 24);
    area_.push_back(2 * 24 * 24);
    area_.push_back(4 * 24 * 24);
    area_.push_back(8 * 24 * 24);
    area_.push_back(16 * 24 * 24);
    area_.push_back(32 * 24 * 24);
    area_.push_back(64 * 24 * 24);

    ratio_.push_back(1);
    sliding_window_stride_ = 16;

    /*   const vector<boost::shared_ptr<Layer<float> > > &layers = net_->layers();
       const vector<vector<Blob<float> *> > &bottom_vecs = net_->bottom_vecs();
       const vector<vector<Blob<float> *> > &top_vecs = net_->top_vecs();
       for (int i = 0; i < layers.size(); ++i) {
           layers[i]->Forward(bottom_vecs[i], top_vecs[i]);
       }*/
}

FaceCaffeDetector::~FaceCaffeDetector() {

}

void FaceCaffeDetector::Forward(vector<cv::Mat> &imgs, vector<vector<Detection> > &final_vbbox) {
    if (imgs.size() == 0)
        return;
    resize_ratios_.clear();

    final_vbbox.resize(imgs.size());

    int scale_num = area_.size() * ratio_.size();
    int max_col = 0, max_row = 0;
    vector<pair<int, int> > addeds;
    vector<pair<int, int> > originImageSize;
    for (int i = 0; i < imgs.size(); i++) {
        originImageSize.push_back(make_pair(imgs[i].cols, imgs[i].rows));
        float resize_ratio = ReScaleImage(imgs[i], img_scale_min_, img_scale_max_);
        if (max_col < imgs[i].cols) {
            max_col = imgs[i].cols;
        }
        if (max_row < imgs[i].rows) {
            max_row = imgs[i].rows;
        }

        resize_ratios_.push_back(resize_ratio);
    }

    for (int i = 0; i < imgs.size(); i++) {
        addeds.push_back(CatImg(imgs[i], max_col, max_row));
    }

    Blob<float> *input_blob = net_->input_blobs()[0];
    Size image_size = Size(max_col, max_row);
    vector<int> shape = {static_cast<int>(imgs.size()), 3, image_size.height, image_size.width};
    input_blob->Reshape(shape);
    net_->Reshape();
    float *input_data = input_blob->mutable_cpu_data();
    for (size_t i = 0; i < imgs.size(); i++) {
        Mat sample;
        Mat img = imgs[i];
        // images from the same batch should have the same size
        CheckChannel(img, num_channels_, sample);
        assert(img.rows == image_size.height && img.cols == image_size.width);
        //GenerateSample(num_channels_, img, sample);

        size_t image_off = i * sample.channels() * sample.rows * sample.cols;
        for (int k = 0; k < sample.channels(); k++) {
            size_t channel_off = k * sample.rows * sample.cols;
            for (int row = 0; row < sample.rows; row++) {
                size_t row_off = row * sample.cols;
                for (int col = 0; col < sample.cols; col++) {
                    input_data[image_off + channel_off + row_off + col] =
                        (float(sample.at<uchar>(row, col * sample.channels() + k)) - pixel_means_[k]) / 1;

                }
            }
        }
    }
    net_->ForwardPrefilled();
    if (use_gpu_) {
        cudaDeviceSynchronize();
    }
    auto cls = net_->blob_by_name(layer_name_cls_);
    auto reg = net_->blob_by_name(layer_name_reg_);

    final_vbbox.clear();
    final_vbbox.resize(0);
    final_vbbox.resize(cls->num());

    assert(cls->channels() == scale_num * 2);
    assert(reg->channels() == scale_num * 4);

    assert(cls->height() == reg->height());
    assert(cls->width() == reg->width());

    const float *cls_cpu = cls->cpu_data();
    const float *reg_cpu = reg->cpu_data();


    vector<Detection> vbbox;
    vector<float> gt_ww, gt_hh;
    gt_ww.resize(scale_num);
    gt_hh.resize(scale_num);

    for (size_t i = 0; i < area_.size(); i++) {
        for (size_t j = 0; j < ratio_.size(); j++) {
            int index = i * ratio_.size() + j;
            gt_ww[index] = sqrt(area_[i] * ratio_[j]);
            gt_hh[index] = gt_ww[index] / ratio_[j];
        }
    }
    int cls_index = 0;
    int reg_index = 0;
    for (int img_idx = 0; img_idx < cls->num(); img_idx++) {
        vbbox.resize(0);
        for (int scale_idx = 0; scale_idx < scale_num; scale_idx++) {
            int skip = cls->height() * cls->width();
            for (int h = 0; h < cls->height(); h++) {
                for (int w = 0; w < cls->width(); w++) {
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

                        for (int j = 0; j < 4; j++) {
                            rect[j] = reg_cpu[reg_index + j * skip];
                        }

                        float shift_x = w * sliding_window_stride_ + sliding_window_stride_ / 2.f - 1;
                        float shift_y = h * sliding_window_stride_ + sliding_window_stride_ / 2.f - 1;
                        rect[2] = exp(rect[2]) * gt_ww[scale_idx];
                        rect[3] = exp(rect[3]) * gt_hh[scale_idx];
                        rect[0] = rect[0] * gt_ww[scale_idx] - rect[2] / 2.f + shift_x;
                        rect[1] = rect[1] * gt_hh[scale_idx] - rect[3] / 2.f + shift_y;
                        if (rect[0] > imgs[img_idx].cols)
                            continue;
                        if (rect[1] > imgs[img_idx].rows)
                            continue;
                        Detection bbox;
                        bbox.confidence = confidence;
                        bbox.box = Rect(rect[0], rect[1], rect[2], rect[3]);
                        bbox.box &= Rect(0, 0, imgs[img_idx].cols, imgs[img_idx].rows);
                        bbox.box &= Rect(0, 0, originImageSize[img_idx].first, originImageSize[img_idx].second);
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

        NMS(vbbox, 0.3);
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

void FaceCaffeDetector::NMS(vector<Detection> &p, float threshold) {
    sort(p.begin(), p.end(), mycmp);
    for (size_t i = 0; i < p.size(); ++i) {
        if (p[i].deleted)
            continue;
        for (size_t j = i + 1; j < p.size(); ++j) {

            if (!p[j].deleted) {
                cv::Rect intersect = p[i].box & p[j].box;
                float iou = intersect.area() * 1.0f / (p[j].box.area() + p[i].box.area() - intersect.area());
                if (iou > threshold) {
                    p[j].deleted = true;
                }
            }
        }
    }
}

int FaceCaffeDetector::Detect(vector<cv::Mat> &imgs,
                              vector<vector<Detection> > &boxes) {
    if (!device_setted_) {
        Caffe::SetDevice(gpu_id_);
        Caffe::set_mode(Caffe::GPU);
        device_setted_ = true;
    }

    //  vector<Blob<float> *> outputs;

    Forward(imgs, boxes);

    // GetDetection(outputs, boxes, imgs);

    return 1;
}

void FaceCaffeDetector::GetDetection(vector<Blob<float> *> &outputs,
                                     vector<vector<Detection> > &final_vbbox, vector<cv::Mat> &imgs) {

}

} /* namespace dg */
