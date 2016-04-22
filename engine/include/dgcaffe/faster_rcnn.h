#ifndef FASTER_RCNN_H_INCLUDED
#define FASTER_RCNN_H_INCLUDED

#include <string>
#include <opencv2/opencv.hpp>
#include <caffe/caffe.hpp>
using namespace std;
using namespace cv;
using namespace caffe;

struct Bbox {
    float confidence;
    Rect rect;
    bool deleted;
    int cls_id;
};

class Faster_rcnn {
 public:
    Faster_rcnn(const string& model_file, const string& trained_file,
                const string& layer_name_rois, const string& layer_name_score,
                const string& layer_name_bbox, const bool use_GPU,
                const int batch_size, const int scale, const float conf_thres,
                const int max_per_img);

    void forward(vector<Mat> imgs, vector<Blob<float>*> &outputs);
    void get_detection(vector<Blob<float>*>& outputs,
                       vector<struct Bbox> &final_vbbox);

 private:
    void nms(vector<struct Bbox>& p, float threshold);
    caffe::shared_ptr<Net<float> > net_;
    int num_channels_;
    int batch_size_;
    bool useGPU_;
    vector<float> pixel_means_;
    int scale_;
    float conf_thres_;
    string layer_name_rois_;
    string layer_name_score_;
    string layer_name_bbox_;
    int sliding_window_stride_;
    int max_per_img_;
};

bool mycmp(struct Bbox b1, struct Bbox b2) {
    return b1.confidence > b2.confidence;
}

Faster_rcnn::Faster_rcnn(const string& model_file, const string& trained_file,
                         const string& layer_name_rois,
                         const string& layer_name_score,
                         const string& layer_name_bbox, const bool use_GPU,
                         const int batch_size, const int scale,
                         const float conf_thres, const int max_per_img) {
    if (use_GPU) {
        Caffe::set_mode(Caffe::GPU);
        Caffe::SetDevice(0);
        useGPU_ = true;
    } else {
        Caffe::set_mode(Caffe::CPU);
        useGPU_ = false;
    }

    /* Set batchsize */
    batch_size_ = batch_size;

    /* Load the network. */
    cout << "loading " << model_file << endl;
    net_.reset(new Net<float>(model_file, TEST));
    cout << "loading " << trained_file << endl;
    net_->CopyTrainedLayersFrom(trained_file);
    CHECK_EQ(net_->num_inputs(), 2)<< "Network should have exactly two input.";

    Blob<float>* input_layer = net_->input_blobs()[0];
    Blob<float>* im_info_layer = net_->input_blobs()[1];
    do {
        vector<int> shape = input_layer->shape();
        shape[0] = batch_size_;
        input_layer->Reshape(shape);
        vector<int> shape_im_info;
        shape_im_info.push_back(batch_size_);
        shape_im_info.push_back(3);
        im_info_layer->Reshape(shape_im_info);
        //net_->Reshape();
    } while (0);

    num_channels_ = input_layer->channels();
    CHECK(num_channels_ == 3 || num_channels_ == 1)
                                                             << "Input layer should have 1 or 3 channels.";
    pixel_means_.push_back(102.9801);
    pixel_means_.push_back(115.9465);
    pixel_means_.push_back(122.7717);
    scale_ = scale;
    conf_thres_ = conf_thres;
    max_per_img_ = max_per_img;
    layer_name_rois_ = layer_name_rois;
    layer_name_score_ = layer_name_score;
    layer_name_bbox_ = layer_name_bbox;
    sliding_window_stride_ = 16;
}

// predict single frame forward function
void Faster_rcnn::forward(vector<cv::Mat> imgs, vector<Blob<float>*> &outputs) {

    Blob<float>* input_layer = net_->input_blobs()[0];
    Blob<float>* im_info_layer = net_->input_blobs()[1];
    float* im_info = im_info_layer->mutable_cpu_data();

    assert(static_cast<int>(imgs.size()) <= batch_size_);
    //resize_shapes.resize(0);
    int cnt = 0;
    assert(imgs.size() == 1);
    for (size_t i = 0; i < imgs.size(); i++) {
        Mat sample;
        Mat img = imgs[i];

        float resize_ratio = 1;
        Size resize_r_c;
        if (img.rows > scale_ && img.cols > scale_) {
            if (img.rows < img.cols) {
                resize_ratio = float(scale_) / img.rows;
                resize_r_c = Size(img.cols * resize_ratio, scale_);
                resize(img, img, resize_r_c);
            } else {
                resize_ratio = float(scale_) / img.cols;
                resize_r_c = Size(scale_, img.rows * resize_ratio);
                resize(img, img, resize_r_c);
            }
            //cvtColor(img, img, CV_RGB2BGR);
        }
        //cout << img.rows << img.cols << endl;

        do {
            vector<int> shape;
            shape.push_back(batch_size_);
            shape.push_back(3);
            shape.push_back(img.rows);
            shape.push_back(img.cols);
            input_layer->Reshape(shape);
            net_->Reshape();
        } while (0);
        //resize_shapes.push_back(resize_r_c);

        if (img.channels() == 3 && num_channels_ == 1)
            cvtColor(img, sample, CV_BGR2GRAY);
        else if (img.channels() == 4 && num_channels_ == 1)
            cvtColor(img, sample, CV_BGRA2GRAY);
        else if (img.channels() == 4 && num_channels_ == 3)
            cvtColor(img, sample, CV_RGBA2BGR);
        else if (img.channels() == 1 && num_channels_ == 3)
            cvtColor(img, sample, CV_GRAY2BGR);
        else
            sample = img;

        float* input_data = input_layer->mutable_cpu_data();

        for (int k = 0; k < sample.channels(); k++) {
            for (int i = 0; i < sample.rows; i++) {
                for (int j = 0; j < sample.cols; j++) {
                    input_data[cnt] = float(sample.at < uchar > (i, j * 3 + k))
                            - pixel_means_[k];
                    cnt += 1;
                }
            }
        }
        im_info[i * 3] = img.rows;
        im_info[i * 3 + 1] = img.cols;
        im_info[i * 3 + 2] = resize_ratio;
    }
    if (useGPU_) {
        cudaDeviceSynchronize();
    }
    net_->ForwardPrefilled();
    if (useGPU_) {
        cudaDeviceSynchronize();
    }

    outputs.resize(0);
    Blob<float>* output_rois = net_->blob_by_name(layer_name_rois_).get();
    Blob<float>* output_score = net_->blob_by_name(layer_name_score_).get();
    Blob<float>* output_bbox = net_->blob_by_name(layer_name_bbox_).get();
    outputs.push_back(output_rois);
    outputs.push_back(output_score);
    outputs.push_back(output_bbox);
}

void Faster_rcnn::nms(vector<struct Bbox>& p, float threshold) {

    sort(p.begin(), p.end(), mycmp);
    int cnt = 0;
    for (size_t i = 0; i < p.size(); i++) {

        if (p[i].deleted)
            continue;
        cnt += 1;
        for (size_t j = i + 1; j < p.size(); j++) {

            if (!p[j].deleted && p[i].cls_id == p[j].cls_id) {
                cv::Rect intersect = p[i].rect & p[j].rect;
                //float iou = intersect.area() * 1.0/p[j].rect.area(); /// (p[i].rect.area() + p[j].rect.area() - intersect.area());
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

void bbox_transform_inv_clip(Blob<float>* roi, Blob<float>* cls,
                             Blob<float>* reg, Blob<float>* im_info_layer,
                             vector<struct Bbox> &vbbox) {
    const float* roi_cpu = roi->cpu_data();
    const float* cls_cpu = cls->cpu_data();
    const float* reg_cpu = reg->cpu_data();
    const float* im_info = im_info_layer->cpu_data();

    vbbox.resize(roi->shape()[0]);
    for (int i = 0; i < roi->shape()[0]; i++) {
        if (i == 0) {
            cout << "0 " << roi_cpu[0] << endl;
            cout << "1 " << roi_cpu[1] << endl;
            cout << "2 " << roi_cpu[2] << endl;
            cout << "3 " << roi_cpu[3] << endl;
            cout << "4 " << roi_cpu[4] << endl;
        }
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

        const float* cur_reg = reg_cpu + i * 20 + 4 * cls_id;
        const float* cur_roi = roi_cpu + i * 5 + 1;
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

        // clip_boxes
        //int x1 = std::max(std::min(float(pred_ctr_x - 0.5 * pred_w), float(im_info[1] -1)),float(0));
        //int y1 = std::max(std::min(float(pred_ctr_y - 0.5 * pred_h), float(im_info[0] -1)),float(0));
        //int x2 = std::max(std::min(float(pred_ctr_x + 0.5 * pred_w), float(im_info[1] -1)),float(0));
        //int y2 = std::max(std::min(float(pred_ctr_y + 0.5 * pred_h), float(im_info[0] -1)),float(0));
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

void Faster_rcnn::get_detection(vector<Blob<float>*>& outputs,
                                vector<struct Bbox> &final_vbbox) {
    Blob<float>* roi = outputs[0];
    Blob<float>* cls = outputs[1];
    Blob<float>* reg = outputs[2];

    assert(roi->shape()[1] == 5);
    assert(cls->num() == reg->num());
    assert(cls->channels() == 5);
    assert(reg->channels() == 20);
    assert(cls->height() == reg->height());
    assert(cls->width() == reg->width());

    vector<struct Bbox> vbbox;

    Blob<float>* im_info_layer = net_->input_blobs()[1];
    const float* im_info = im_info_layer->cpu_data();
    bbox_transform_inv_clip(roi, cls, reg, im_info_layer, vbbox);
    float resize_ratio = im_info[2];

    if (vbbox.size() != 0) {

        sort(vbbox.begin(), vbbox.end(), mycmp);
        vbbox.resize(min(static_cast<size_t>(max_per_img_), vbbox.size()));
        nms(vbbox, 0.2);
    }

    final_vbbox.resize(0);
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

#endif // FASTER_RCNN_H_INCLUDED

