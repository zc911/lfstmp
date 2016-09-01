#include "vehicle_multi_type_detector.h"
#include "alg/caffe_helper.h"

namespace dg {

static bool mycmp(struct Bbox b1, struct Bbox b2) {
    return b1.confidence > b2.confidence;
}

VehicleMultiTypeDetector::VehicleMultiTypeDetector(
    const VehicleMultiTypeConfig &config)
    : config_(config) {

    if (config.use_gpu) {
        Caffe::SetDevice(config.gpu_id);
        Caffe::set_mode(Caffe::GPU);
    } else {
        Caffe::set_mode(Caffe::CPU);
    }

    // currently, this detector only support batch size = 1
    batch_size_ = config_.batch_size = 1;
    scale_ = config_.target_min_size;

    string deploy_content;
        ModelsMap *modelsMap = ModelsMap::GetInstance();

    modelsMap->getModelContent(config.deploy_file,deploy_content);
    net_.reset(
        new Net<float>(config.deploy_file,deploy_content,TEST));
    string model_content;
    modelsMap->getModelContent(config.model_file,model_content);
        net_->CopyTrainedLayersFrom(config.model_file,model_content);

    CHECK_EQ(net_->num_inputs(), 2) << "Network should have exactly two input.";

    Blob<float> *input_layer = net_->input_blobs()[0];
    Blob<float> *im_info_layer = net_->input_blobs()[1];

    vector<int> shape = input_layer->shape();
    shape[0] = batch_size_;
    input_layer->Reshape(shape);
    vector<int> shape_im_info;
    shape_im_info.push_back(batch_size_);
    shape_im_info.push_back(3);
    im_info_layer->Reshape(shape_im_info);

    num_channels_ = input_layer->channels();
    CHECK(num_channels_ == 3 || num_channels_ == 1)
    << "Input layer should have 1 or 3 channels.";
    pixel_means_.push_back(102.9801);
    pixel_means_.push_back(115.9465);
    pixel_means_.push_back(122.7717);

    conf_thres_ = 0.5;
    max_per_img_ = 100;
    layer_name_rois_ = "rois";
    layer_name_score_ = "cls_prob";

    layer_name_bbox_ = "bbox_pred";
    sliding_window_stride_ = 16;
    net_->Reshape();
 /*   const vector<boost::shared_ptr<Layer<float> > > &layers = net_->layers();
    const vector<vector<Blob<float> *> > &bottom_vecs = net_->bottom_vecs();
    const vector<vector<Blob<float> *> > &top_vecs = net_->top_vecs();
    for (int i = 0; i < layers.size(); ++i) {
        layers[i]->Forward(bottom_vecs[i], top_vecs[i]);
    }*/
}

VehicleMultiTypeDetector::~VehicleMultiTypeDetector() {

}

vector<Detection> VehicleMultiTypeDetector::Detect(const cv::Mat &img) {

    vector<Mat> images;
    vector<Blob<float> *> tmp_outputs;
    vector<struct Bbox> tmp_result;
    vector<Detection> result;
    images.push_back(img);
    forward(images, tmp_outputs);

    getDetection(tmp_outputs, tmp_result);
    for (int i = 0; i < tmp_result.size(); ++i) {
        Bbox bbox = tmp_result[i];
        Detection detection;
        detection.box = bbox.rect;
        detection.confidence = bbox.confidence;
        detection.id = bbox.cls_id;
        result.push_back(detection);
    }

    return result;

}

vector<vector<Detection> > VehicleMultiTypeDetector::DetectBatch(

    const vector<cv::Mat> &img) {
    vector<vector<Detection>> results;

    for (int i = 0; i < img.size(); ++i) {
        vector<Detection> detections = Detect(img[i]);
        results.push_back(detections);
    }

    return results;
}

// predict single frame forward function
void VehicleMultiTypeDetector::forward(vector<cv::Mat> imgs,
                                       vector<Blob<float> *> &outputs) {

    Blob<float> *input_layer = net_->input_blobs()[0];
    Blob<float> *im_info_layer = net_->input_blobs()[1];
    float *im_info = im_info_layer->mutable_cpu_data();

    int cnt = 0;

    for (size_t i = 0; i < imgs.size(); i++) {
        Mat sample;
        Mat img = imgs[i];

        float resize_ratio = ReScaleImage(img, scale_);
        vector<int> shape;
        shape.push_back(batch_size_);
        shape.push_back(3);
        shape.push_back(img.rows);
        shape.push_back(img.cols);
        input_layer->Reshape(shape);
        net_->Reshape();

        CheckChannel(img, num_channels_, sample);

        float *input_data = input_layer->mutable_cpu_data();

        for (int k = 0; k < sample.channels(); k++) {
            for (int i = 0; i < sample.rows; i++) {
                for (int j = 0; j < sample.cols; j++) {
                    input_data[cnt] = float(sample.at<uchar>(i, j * 3 + k))
                        - pixel_means_[k];
                    cnt += 1;
                }
            }
        }
        im_info[i * 3] = img.rows;
        im_info[i * 3 + 1] = img.cols;
        im_info[i * 3 + 2] = resize_ratio;
    }

    net_->ForwardPrefilled();

    if (config_.use_gpu) {
        cudaDeviceSynchronize();
    }

    outputs.resize(0);
    Blob<float> *output_rois = net_->blob_by_name(layer_name_rois_).get();
    Blob<float> *output_score = net_->blob_by_name(layer_name_score_).get();
    Blob<float> *output_bbox = net_->blob_by_name(layer_name_bbox_).get();
    outputs.push_back(output_rois);
    outputs.push_back(output_score);
    outputs.push_back(output_bbox);
}

void VehicleMultiTypeDetector::getDetection(vector<Blob<float> *> &outputs,
                                            vector<struct Bbox> &final_vbbox) {
    Blob<float> *roi = outputs[0];
    Blob<float> *cls = outputs[1];
    Blob<float> *reg = outputs[2];

    assert(roi->shape()[1] == 5);
    assert(cls->num() == reg->num());
    assert(cls->channels() == 5);
    assert(reg->channels() == 20);
    assert(cls->height() == reg->height());
    assert(cls->width() == reg->width());

    vector<struct Bbox> vbbox;

    Blob<float> *im_info_layer = net_->input_blobs()[1];
    const float *im_info = im_info_layer->cpu_data();
    bboxTransformInvClip(roi, cls, reg, im_info_layer, vbbox);
    float resize_ratio = im_info[2];

    if (vbbox.size() != 0) {

        sort(vbbox.begin(), vbbox.end(), mycmp);
        vbbox.resize(min(static_cast<size_t>(max_per_img_), vbbox.size()));
        nms(vbbox, 0.2);
    }
    final_vbbox.clear();
    for (size_t i = 0; i < vbbox.size(); i++) {

        if (!vbbox[i].deleted) {
            struct Bbox box = vbbox[i];
            float x = box.rect.x / resize_ratio;
            float y = box.rect.y / resize_ratio;
            float w = box.rect.width / resize_ratio;
            float h = box.rect.height / resize_ratio;
            box.rect.x = x;
            box.rect.y = y;
            box.rect.width = w;
            box.rect.height = h;
            if (vbbox[i].confidence > conf_thres_) {
                final_vbbox.push_back(box);
            }
        }
    }
}

void VehicleMultiTypeDetector::nms(vector<struct Bbox> &p, float threshold) {

    sort(p.begin(), p.end(), mycmp);
    int cnt = 0;
    for (size_t i = 0; i < p.size(); i++) {

        if (p[i].deleted)
            continue;
        cnt += 1;
        for (size_t j = i + 1; j < p.size(); j++) {

            if (!p[j].deleted && p[i].cls_id == p[j].cls_id) {
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
}

void VehicleMultiTypeDetector::bboxTransformInvClip(
    Blob<float> *roi, Blob<float> *cls, Blob<float> *reg,
    Blob<float> *im_info_layer, vector<struct Bbox> &vbbox) {
    const float *roi_cpu = roi->cpu_data();
    const float *cls_cpu = cls->cpu_data();
    const float *reg_cpu = reg->cpu_data();
    const float *im_info = im_info_layer->cpu_data();

    vbbox.resize(roi->shape()[0]);
    for (int i = 0; i < roi->shape()[0]; i++) {
        int cls_id = 0;
        float prob = 0;
        for (int li = 0; li < 5; li++) {
            if (prob < cls_cpu[5 * i + li]) {
                prob = cls_cpu[5 * i + li];
                cls_id = li;
            }
        }
        int width, height, ctr_x, ctr_y;
        float dx, dy, dw, dh, pred_ctr_x, pred_ctr_y, pred_w, pred_h;

        const float *cur_reg = reg_cpu + i * 20 + 4 * cls_id;
        const float *cur_roi = roi_cpu + i * 5 + 1;
        width = cur_roi[2] - cur_roi[0] + 1.0;
        height = cur_roi[3] - cur_roi[1] + 1.0;
        ctr_x = cur_roi[0] + 0.5 * width;
        ctr_y = cur_roi[1] + 0.5 * height;
        dx = cur_reg[0];
        dy = cur_reg[1];
        dw = cur_reg[2];
        dh = cur_reg[3];
        pred_ctr_x = dx * width + ctr_x;
        pred_ctr_y = dy * height + ctr_y;
        pred_w = exp(dw) * width;
        pred_h = exp(dh) * height;

        struct Bbox &bbox = vbbox[i];

        if (cls_id >= 1) {
            bbox.confidence = prob;
            bbox.cls_id = cls_id;
        } else {
            bbox.confidence = 1.0 - prob;
        }
        bbox.rect = Rect(pred_ctr_x - 0.5 * pred_w, pred_ctr_y - 0.5 * pred_h,
                         pred_w, pred_h)
            & Rect(0, 0, im_info[1] - 1, im_info[0] - 1);
        bbox.deleted = false;
    }
}

}
