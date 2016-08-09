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
    input_geometry_ = cv::Size(input_layer->width(),input_layer->height());
    /*input_geometry_ = cv::Size(config.target_max_size, config.target_min_size);
    input_layer->Reshape(batch_size_, num_channels_,
                         input_geometry_.height,
                         input_geometry_.width);
    net_->Reshape();*/

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

void MarkerCaffeSsdDetector::Fullfil(vector<cv::Mat> &images_tiny,
                                     vector<Blob<float> *> &outputs,
                                     vector<vector<Detection> > &detect_results,
                                     vector<vector<Rect> > &fobs,
                                     vector<vector<float> > &params) {

    int image_offset = detect_results.size();
    for (int i = 0; i < images_tiny.size(); ++i) {
        vector<Detection> imageDetection;
        detect_results.push_back(imageDetection);
    }
    int box_num = outputs[0]->height();
    const float* top_data = outputs[0]->cpu_data();
    vector<float> crop_xmin=params[0];
    vector<float> crop_ymin=params[1];
    vector<float> thresh_ymin=params[2];
    vector<float> thresh_ymax=params[3];
    vector<float> row_ratio = params[4];
    vector<float> col_ratio = params[5];
    float cls_conf[7] = {1.0, 0.36, 0.6, 0.6, 0.5, 0.6, 0.6};
    for(int j = 0; j < box_num; j++) {
        int img_id = top_data[j * 7 + 0];
        if (img_id < 0 || img_id >= detect_results.size()) {
            LOG(ERROR) << "Image id invalid: " << img_id << endl;
            continue;
        }
        vector<Detection> &imageDetection = detect_results[image_offset + img_id];

        int cls = top_data[j * 7 + 1];
        float score = top_data[j * 7 + 2];
        float xmin = top_data[j * 7 + 3] * images_tiny[img_id].cols;
        float ymin = top_data[j * 7 + 4] * images_tiny[img_id].rows;
        float xmax = top_data[j * 7 + 5] * images_tiny[img_id].cols;
        float ymax = top_data[j * 7 + 6] * images_tiny[img_id].rows;
        if (score > cls_conf[cls]) {

            xmin *= col_ratio[img_id];
            xmax *= col_ratio[img_id];
            ymin *= row_ratio[img_id];
            ymax *= row_ratio[img_id];

            xmin += crop_xmin[img_id];
            xmax += crop_xmin[img_id];
            ymin += crop_ymin[img_id];
            ymax += crop_ymin[img_id];
            // exclude bboxes that lie outside car window.
            if ((thresh_ymin[img_id]-ymin)/(ymax-ymin) > 0.3 ||
                (ymax-thresh_ymax[img_id])/(ymax-ymin) > 0.3)
                continue;

            // exclude bboxes that lie in a predefined fobbiden place

            vector<Rect> fob = fobs[img_id];

            Rect overlap = fob[cls] & Rect(xmin,ymin,xmax-xmin,ymax-ymin);
            if (int(overlap.area()) > 1) {
                continue;  //exclude this box
            }
            Detection detection;
            detection.box =  Rect(xmin,ymin,xmax-xmin,ymax-ymin);
            detection.id = cls;
            detection.confidence = score;
            imageDetection.push_back(detection);

        }
    }

}
int MarkerCaffeSsdDetector::DetectBatch(vector<cv::Mat> &imgs, vector<vector<Detection> > &window_detections,
                                        vector<vector<Detection> > &detect_results) {
    float costtime, diff;

    struct timeval start, end;
    gettimeofday(&start, NULL);

    if (!device_setted_) {
        Caffe::SetDevice(gpu_id_);
        Caffe::set_mode(Caffe::GPU);
        device_setted_ = true;
    }


    detect_results.clear();
    vector<cv::Mat> toPredict;
        vector<cv::Mat> origins;

    vector<vector<float> > params;
    params.resize(6);
    vector<vector<Rect> > fobs;
    for (int i = 0; i < imgs.size(); ++i) {
        cv::Mat image = imgs[i].clone();

        // fobbiden areas
        int xmin,ymin,xmax,ymax;
        if(window_detections[i].size()>0) {
            xmin = window_detections[i][0].box.x;
            ymin = window_detections[i][0].box.y;
            xmax = window_detections[i][0].box.x + window_detections[i][0].box.width;
            ymax = window_detections[i][0].box.y + window_detections[i][0].box.height;
        }else{
            xmin=ymin=0;
            xmax=ymax=1;
        }
            vector<Rect> fob = forbidden_area(xmin, ymin, xmax, ymax);
            fobs.push_back(fob);

            // crop and resize image for tiny.
            int cxmin, cymin;
            Mat img = crop_image(image, xmin, ymin, xmax, ymax, &cxmin, &cymin);
            params[0].push_back(cxmin);
            params[1].push_back(cymin);

            // obtain y threshold for detected object that lies outside car window
            int tymin;
            int tymax;
            float ratio = 0.15;
            show_enlarged_box(image, xmin, ymin, xmax, ymax, &tymin, &tymax, ratio);
            params[2].push_back(tymin);
            params[3].push_back(tymax);

            float target_row = 256;
            float target_col = 384;
            params[4].push_back(img.rows * 1.0 / target_row);
            params[5].push_back(img.cols * 1.0 / target_col);

            // only process images that has a car window.
            // only count images that has a car window.
            if(img.rows>0&&img.cols>0){
                resize(img, img, Size(target_col, target_row));
            }else{
                img=Mat::zeros(Size(target_col, target_row),CV_8UC3);
            }

            toPredict.push_back(img);
        
        if (toPredict.size() == batch_size_) {

            vector<Blob<float> *> outputs = PredictBatch(toPredict);

            Fullfil(toPredict, outputs, detect_results, fobs, params);

            toPredict.clear();
        }
    }

    if (toPredict.size() > 0) {
        vector<Blob<float> *> outputs = PredictBatch(toPredict);
        Fullfil(toPredict, outputs, detect_results, fobs, params);
    }
        gettimeofday(&end, NULL);

        diff = ((end.tv_sec - start.tv_sec) * 1000000 + end.tv_usec - start.tv_usec)
            / 1000.f;
            LOG(INFO)<<"window marker  batch "<<diff;

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

std::vector<Blob<float> *> MarkerCaffeSsdDetector::PredictBatch(const vector<Mat> &imgs) {
    float costtime, diff;
    struct timeval start, end;
    gettimeofday(&start, NULL);
    vector<Blob<float> *> outputs;

    Blob<float> *input_layer = net_->input_blobs()[0];
    input_geometry_.height=imgs[0].rows;
    input_geometry_.width=imgs[0].cols;
    if (imgs.size() <= batch_size_) {
        input_layer->Reshape(imgs.size(), num_channels_,
                             input_geometry_.height,
                             input_geometry_.width);
        net_->Reshape();
    } else {
        LOG(ERROR) << "Input images size is more than batch size!" << endl;
        return outputs;
    }
    float *input_data = input_layer->mutable_cpu_data();

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
    net_->Reshape();
    net_->ForwardPrefilled();

    if (use_gpu_) {
        cudaDeviceSynchronize();
    }


    for (int i = 0; i < net_->num_outputs(); i++) {
        Blob<float> *output_layer = net_->output_blobs()[i];
        outputs.push_back(output_layer);
    }
        gettimeofday(&end, NULL);

            diff = ((end.tv_sec - start.tv_sec) * 1000000 + end.tv_usec - start.tv_usec)
            / 1000.f;
            LOG(INFO)<<"marker predict batch "<<diff;
    return outputs;
}

}