//
// Created by jiajaichen on 16-8-5.
//

#include "marker_caffe_ssd_detector.h"
#include "vehicle_caffe_detector.h"
#include "alg/caffe_helper.h"

namespace dg {
MarkerCaffeSsdDetector::MarkerCaffeSsdDetector(const VehicleCaffeDetectorConfig &config) : caffe_config_(config) {

    use_gpu_ = config.use_gpu;

    gpu_id_ = config.gpu_id;
    threshold_ = config.threshold;

    if (use_gpu_) {

        Caffe::SetDevice(gpu_id_);
        Caffe::set_mode(Caffe::GPU);

        use_gpu_ = true;
    }
    else {
        Caffe::set_mode(Caffe::CPU);
        use_gpu_ = false;

    }

    batch_size_ = config.batch_size;
    //  net_.reset(new Net<float>(config.deploy_file, TEST));

#if DEBUG
    net_.reset(
        new Net<float>(config.deploy_file, TEST));
#else
    net_.reset(
            new Net<float>(config.deploy_file, TEST, config.is_model_encrypt));
#endif

    net_->CopyTrainedLayersFrom(config.model_file);

    Blob<float> *input_layer = net_->input_blobs()[0];
    num_channels_ = input_layer->channels();
    input_geometry_ = cv::Size(config.target_max_size, config.target_min_size);
    input_layer->Reshape(batch_size_, num_channels_,
                         input_geometry_.height,
                         input_geometry_.width);
    net_->Reshape();

/*    const vector<boost::shared_ptr<Layer<float> > > &layers = net_->layers();
    const vector<vector<Blob<float> *> > &bottom_vecs = net_->bottom_vecs();
    const vector<vector<Blob<float> *> > &top_vecs = net_->top_vecs();
    for (int i = 0; i < layers.size(); ++i) {
        layers[i]->Forward(bottom_vecs[i], top_vecs[i]);
    }
*/
    device_setted_ = false;
#ifdef SHOW_VIS
    color_.push_back(Scalar(255, 0, 0));
    color_.push_back(Scalar(0, 255, 0));
    color_.push_back(Scalar(0, 0, 255));
    color_.push_back(Scalar(255, 255, 0));
    color_.push_back(Scalar(0, 255, 255));
    color_.push_back(Scalar(255, 0, 255));
    tags_.push_back("bg");
    tags_.push_back("car");
    tags_.push_back("person");
    tags_.push_back("bicycle");
    tags_.push_back("tricycle");
#endif

}

MarkerCaffeSsdDetector::~MarkerCaffeSsdDetector() {

}

void MarkerCaffeSsdDetector::Fullfil(vector<cv::Mat> &img,
                                   vector<Blob<float> *> &outputs,
vector<vector<Detection> > &detect_results,vector<vector<Rect> > fobs,vector<vector<float> > params) {
int tot_cnt = 0;
int box_num = outputs[tot_cnt]->height();

const float *top_data = outputs[tot_cnt]->cpu_data();

int image_offset = detect_results.size();
for (int i = 0; i < img.size(); ++i) {
vector<Detection> imageDetection;
detect_results.push_back(imageDetection);
}

for (int j = 0; j < box_num; j++) {
int img_id = top_data[j * 7 + 0];
if (img_id < 0 || img_id >= detect_results.size()) {
LOG(ERROR) << "Image id invalid: " << img_id << endl;
continue;
}
vector<Detection> &imageDetection = detect_results[image_offset + img_id];

int cls = top_data[j * 7 + 1];
float score = top_data[j * 7 + 2];
float xmin = top_data[j * 7 + 3] * img[img_id].cols;
float ymin = top_data[j * 7 + 4] * img[img_id].rows;
float xmax = top_data[j * 7 + 5] * img[img_id].cols;
float ymax = top_data[j * 7 + 6] * img[img_id].rows;

if (score > threshold_) {

Detection detection;
detection.box = Rect(xmin, ymin, xmax - xmin, ymax - ymin);
detection.id = cls;
detection.confidence = score;
imageDetection.push_back(detection);

#ifdef SHOW_VIS
char char_score[100];
            sprintf(char_score, "%.3f", score);
            rectangle(img[img_id], Rect(xmin, ymin, xmax - xmin, ymax - ymin), color_[cls]);
            putText(img[img_id],
                    tags_[cls] + "_" + string(char_score),
                    Point(xmin, ymin),
                    CV_FONT_HERSHEY_COMPLEX,
                    0.5,
                    color_[0]);
#endif

}
}
#ifdef  SHOW_VIS
for(int i = 0; i < img.size(); ++i){
        cv::Mat image = img[i];
        imshow("debug.jpg", image);
        waitKey(-1);
    }
#endif
}
int MarkerCaffeSsdDetector::DetectBatch(vector<cv::Mat> &img,vector<Detection> &window_detections,
                                      vector<vector<Detection> > &detect_results) {

    if (!device_setted_) {
        Caffe::SetDevice(gpu_id_);
        Caffe::set_mode(Caffe::GPU);
        device_setted_ = true;
    }


    detect_results.clear();
    vector<cv::Mat> toPredict;
    vector<float> row_ratio;
    vector<float> col_ratio;
    vector<float> crop_xmin;
    vector<float> crop_ymin;
    vector<float> thresh_ymin;
    vector<float> thresh_ymax;
    vector< vector<float> > params;
    params.resize(6);
    vector<vector<Rect> > fobs;
    for (int i = 0; i < img.size(); ++i) {
        cv::Mat image = img[i];

        // fobbiden areas
        int xmin=window_detections[i].box.x;
        int ymin = window_detections[i].box.y;
        int xmax=window_detections[i].box.x+window_detections[i].box.width;
        int ymax=window_detections[i].box.y+window_detections[i].box.height;
        vector<Rect> fob = forbidden_area(xmin, ymin, xmax, ymax);
        fobs.push_back(fob);

        // crop and resize image for tiny.
        int cxmin, cymin;
        Mat img = crop_image(images_origin[img_id], xmin, ymin, xmax, ymax, &cxmin, &cymin);
        img_ids.push_back(img_id);
        params[0].push_back(cxmin);
        params[1].push_back(cymin);

        // obtain y threshold for detected object that lies outside car window
        int tymin;
        int tymax;
        float ratio = 0.15;
        show_enlarged_box(images_origin[img_id], xmin, ymin, xmax, ymax, &tymin, &tymax, ratio);
        params[2].push_back(tymin);
        params[3].push_back(tymax);

        float target_row = 256;
        float target_col = 384;
        params[4].push_back(img.rows * 1.0 / target_row);
        params[5].push_back(img.cols * 1.0 / target_col);

        // only process images that has a car window.
        // only count images that has a car window.
        resize(image, image, Size(target_col, target_row));
        toPredict.push_back(image);
        if (toPredict.size() == batch_size_) {

            vector<Blob<float> *> outputs = PredictBatch(toPredict);
            Fullfil(toPredict, outputs, detect_results,fobs,params);
            toPredict.clear();
        }
    }

    if (toPredict.size() > 0) {
        vector<Blob<float> *> outputs = PredictBatch(toPredict);
        Fullfil(toPredict, outputs, detect_results,fobs,params);
    }

//    // make sure batch size is times of the batch size
//    if (img.size() % batch_size_ != 0) {
//        int batchShort = batch_size_ - (img.size() % batch_size_);
//        for (int i = 0; i < batchShort; ++i) {
//            DLOG(INFO) << "Input images size less than batch size" << endl;
//            img.push_back(cv::Mat(1, 1, CV_8UC3));
//        }
//    }
//
//    detect_results.clear();
//    vector<cv::Mat> toPredict;
//    for (int i = 0; i < img.size(); ++i) {
//        cv::Mat image = img[i];
//        toPredict.push_back(image);
//        if (toPredict.size() == batch_size_) {
//            vector<Blob<float> *> outputs = PredictBatch(toPredict);
//            Fullfil(toPredict, outputs, detect_results);
//            toPredict.clear();
//        }
//
//    }

}

std::vector<Blob<float> *> VehicleCaffeDetector::PredictBatch(const vector<Mat> &imgs) {

    vector<Blob<float> *> outputs;

    Blob<float> *input_layer = net_->input_blobs()[0];
    float *input_data = input_layer->mutable_cpu_data();

    if (imgs.size() <= batch_size_) {
        input_layer->Reshape(imgs.size(), num_channels_,
                             input_geometry_.height,
                             input_geometry_.width);
        net_->Reshape();
    } else {
        LOG(ERROR) << "Input images size is more than batch size!" << endl;
        return outputs;
    }
    int cnt = 0;
    DLOG(INFO) << "Start predict batch, size: " << imgs.size() << endl;
    for (int i = 0; i < imgs.size(); i++) {
        cv::Mat sample;
        cv::Mat img = imgs[i];


        GenerateSample(num_channels_, img, sample);

        if ((sample.rows != input_geometry_.height) || (sample.cols != input_geometry_.width)) {
            cv::resize(sample, sample, Size(input_geometry_.width, input_geometry_.height));
        }

        float mean[3] = {104, 117, 123};
        for (int k = 0; k < sample.channels(); k++) {
            for (int i = 0; i < sample.rows; i++) {
                for (int j = 0; j < sample.cols; j++) {
                    input_data[cnt] = sample.at<uchar>(i, j * 3 + k) - mean[k];
                    cnt += 1;
                }
            }
        }
    }

    net_->ForwardPrefilled();

    if (use_gpu_) {
        cudaDeviceSynchronize();
    }

    for (int i = 0; i < net_->num_outputs(); i++) {
        Blob<float> *output_layer = net_->output_blobs()[i];
        outputs.push_back(output_layer);
    }
    return outputs;
}

}