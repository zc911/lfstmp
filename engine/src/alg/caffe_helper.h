/*
 * caffe_helper.h
 *
 *  Created on: Apr 21, 2016
 *      Author: jiajaichen
 */

#ifndef SRC_ALG_CAFFE_HELPER_H_
#define SRC_ALG_CAFFE_HELPER_H_

#include <utility>
#include <iostream>
#include <vector>
#include <algorithm>
#include <opencv2/core/core.hpp>
#include "model/basic.h"
#include "model/model.h"

using namespace cv;
using namespace std;

namespace dg {

struct Bbox {
    float confidence;
    Rect rect;
    bool deleted;
    int cls_id;
};

static void normalize_image(const Mat &input_img, Mat &result) {
    Mat img;
    resize(input_img, img, Size(227, 227));
    result = Mat::zeros(img.rows, img.cols, CV_32FC3);
    float max_val = 0;
    float min_val = 255;
    for (int i = 0; i < img.rows; i++) {
        const uchar *data = img.ptr<uchar>(i);
        for (int j = 0; j < img.cols * 3; j++) {
            if (data[j] > max_val)
                max_val = data[j];
            if (data[j] < min_val)
                min_val = data[j];
        }
    }
    max_val = max_val - min_val;
    if (max_val < 1)
        max_val = 1;
    for (int i = 0; i < img.rows; i++) {
        const uchar *src_data = img.ptr<uchar>(i);
        float *target_data = result.ptr<float>(i);
        for (int j = 0; j < img.cols * 3; j++)
            target_data[j] = (((float) src_data[j]) - min_val) / max_val - 0.5;
    }
}

static vector<vector<Mat> > PrepareBatch(const vector<Mat> &image,
                                         int batch_size) {
    vector<vector<Mat> > vimg;
    vector<Mat> img = image;
    if (img.size() == 0)
        return vimg;

    int padding_size = (batch_size - img.size() % batch_size) % batch_size;
    for (int i = 0; i < padding_size; i++) {
        Mat zero = Mat::zeros(img[0].rows, img[0].cols, CV_8UC3);
        img.push_back(zero);
    }
    int idx = 0;
    while (idx < img.size() / batch_size) {
        auto tmp = img.begin() + idx * batch_size;
        vimg.push_back(vector<Mat>(tmp, tmp + batch_size));
        idx++;
    }

    return vimg;
}

static bool PredictionMoreCmp(const Prediction &b1, const Prediction &b2) {
    return b1.second > b2.second;
}
static bool PredictionLessCmp(const Prediction &b1, const Prediction &b2) {
    return b1.second < b2.second;
}
static void SortPrediction(vector<vector<Prediction> > &dstPreds) {
    for (int i = 0; i < dstPreds.size(); i++) {

        vector<Prediction> dstPred = dstPreds.at(i);
        sort(dstPred.begin(), dstPred.end(), PredictionMoreCmp);
        dstPreds[i] = dstPred;

    }
}

static Prediction MaxPrediction(vector<Prediction> &pre) {
    vector<Prediction>::iterator max = max_element(pre.begin(), pre.end(),
                                                   PredictionLessCmp);
    return *max;
}

static Prediction nthPrediction(vector<Prediction> &pre, int n) {
    if (n > pre.size()) {
        Prediction empty;
        return empty;
    }
    nth_element(pre.begin(), pre.begin() + n, pre.end(),
                PredictionLessCmp);
    return pre.at(n);
}
static bool PairCompare(const std::pair<float, int> &lhs,
                        const std::pair<float, int> &rhs) {
    return lhs.first > rhs.first;
}

static std::vector<int> Argmax(const std::vector<float> &v, int N) {
    std::vector<std::pair<float, int> > pairs;
    for (size_t i = 0; i < v.size(); ++i)
        pairs.push_back(std::make_pair(v[i], i));
    std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(),
                      PairCompare);

    std::vector<int> result;
    for (int i = 0; i < N; ++i)
        result.push_back(pairs[i].second);
    return result;
}

static bool detectionCmp(Detection b1, Detection b2) {
    return b1.confidence > b2.confidence;
}

static void detectionNMS(vector<Detection> &p, float threshold) {
    sort(p.begin(), p.end(), detectionCmp);
    int cnt = 0;
    for (int i = 0; i < p.size(); i++) {
        if (p[i].deleted)
            continue;
        cnt += 1;
        for (int j = i + 1; j < p.size(); j++) {
            if (!p[j].deleted) {
                cv::Rect intersect = p[i].box & p[j].box;
                float iou =
                    intersect.area() * 1.0
                        / (p[i].box.area() + p[j].box.area()
                            - intersect.area());
                if (iou > threshold) {
                    p[j].deleted = true;
                }
                if (intersect.x >= p[i].box.x - 0.2
                    && intersect.y >= p[i].box.y - 0.2
                    && (intersect.x + intersect.width)
                        <= (p[i].box.x + p[i].box.width + 0.2)
                    && (intersect.y + intersect.height)
                        <= (p[i].box.y + p[i].box.height + 0.2)) {
                    p[j].deleted = true;

                }
            }
        }
    }
}

static bool BboxCmp(struct Bbox b1, struct Bbox b2) {
    return b1.confidence > b2.confidence;
}

static void NMS(vector<struct Bbox> &p, float threshold) {
    sort(p.begin(), p.end(), BboxCmp);
    for (size_t i = 0; i < p.size(); ++i) {
        if (p[i].deleted)
            continue;
        for (size_t j = i + 1; j < p.size(); ++j) {

            if (!p[j].deleted) {
                cv::Rect intersect = p[i].rect & p[j].rect;
                float iou = intersect.area() * 1.0f / p[j].rect.area();
                if (iou > threshold) {
                    p[j].deleted = true;
                }
            }
        }
    }
}

static float ReScaleImage(Mat &img, unsigned int scale) {

    Size resize_r_c;
    float resize_ratio = 1;

    if (img.rows > scale && img.cols > scale) {
        if (img.rows < img.cols) {
            resize_ratio = float(scale) / img.rows;
            resize_r_c = Size(img.cols * resize_ratio, scale);
            resize(img, img, resize_r_c);
        } else {
            resize_ratio = float(scale) / img.cols;
            resize_r_c = Size(scale, img.rows * resize_ratio);
            resize(img, img, resize_r_c);
        }
    }

    return resize_ratio;
}

static void CheckChannel(Mat &img, unsigned char tarChannel, Mat &newImage) {

    if (img.channels() == 3 && tarChannel == 1)
        cvtColor(img, newImage, CV_BGR2GRAY);
    else if (img.channels() == 4 && tarChannel == 1)
        cvtColor(img, newImage, CV_BGRA2GRAY);
    else if (img.channels() == 4 && tarChannel == 3)
        cvtColor(img, newImage, CV_RGBA2BGR);
    else if (img.channels() == 1 && tarChannel == 3)
        cvtColor(img, newImage, CV_GRAY2BGR);
    else
        newImage = img;
}

static void GenerateSample(int num_channels_, cv::Mat &img, cv::Mat &sample) {
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
}

static cv::Mat crop_image(cv::Mat image, float xmin, float ymin, float xmax, float ymax, int* cxmin, int* cymin) {
    Mat img = image.clone();
    int img_width = img.cols;
    int img_height = img.rows;
//	float centerx = (xmin + xmax) / 2.0;
    int centery = (ymin + ymax) / 2;
    int width = abs(xmax - xmin);
//    float height = abs(ymax - ymin);

    int width_add = width / 4;  // add width on one side
    int crop_width = width + width_add * 2;
    int crop_xmin = xmin - width_add;
    int crop_xmax = xmax + width_add;

    int crop_height = crop_width * 2 / 3; // =height/1.5
    int crop_ymin = centery - crop_height / 2;
    int crop_ymax = centery + crop_height / 2;

//    cout << xmin << " "<< xmax << " " << ymin << " "<< ymax << endl;
//    cout << crop_xmin << " "<< crop_xmax << " " << crop_ymin << " "<< crop_ymax << endl;
//    cout << "crop width " << crop_width << " crop height " << crop_height << endl;


    if (crop_width > img_width) {
 //       cout << "hconcat started " << endl;
        crop_xmin = 0;
        crop_xmax = crop_width;
        char cw[100], iw[100], ih[100];
        sprintf(cw, "%.3d", crop_width);
        sprintf(iw, "%.3d", img_width);
        sprintf(ih, "%.3d", img_height);
  //      cout << "crop_width " << String(cw) << endl;
  //      cout << "img width " << String(iw) << "img height " << String(ih) << endl;
        Mat cols = Mat::zeros(img_height, int(crop_width) - img_width + 1, img.type()); // +1 for input > 0
        cv::hconcat(img, cols, img);
    }
    else if (crop_xmin < 0) {
        crop_xmin = 0;
        crop_xmax = crop_width;
    }
    else if (crop_xmax >= img_width) {
        crop_xmax = img_width;
        crop_xmin = img_width - crop_width;
    }
    // the operation above may change the dimension of image
    img_width = img.cols;
    img_height = img.rows;
    if (crop_height > img_height) {
        crop_ymin = 0;
        crop_ymax = crop_height;
   //     cout << "add rows started" << endl;
        Mat rows = Mat::zeros(int(crop_height) - img_height + 1, img_width, img.type()); // +1 for input > 0
        img.push_back(rows);
    }
    else if (crop_ymin < 0) {
        crop_ymin = 0;
        crop_ymax = crop_height;
    }
    else if (crop_ymax >= img_height) {
        crop_ymax = img_height;
        crop_ymin = img_height - crop_height;
    }
    img = img(Rect(crop_xmin, crop_ymin, floor(crop_xmax-crop_xmin), floor(crop_ymax-crop_ymin)));
//    cout << "crop succeed" << endl;
    *cxmin = crop_xmin;
    *cymin = crop_ymin;
    return img;
}


static void show_enlarged_box(cv::Mat tmp,cv::Mat image, float xmin, float ymin, float xmax, float ymax, int* cymin, int* cymax, float ratio) {
    Mat img = image;
    float img_width = img.cols;
    float img_height = img.rows;
//	float centerx = (xmin + xmax) / 2.0;
    float centery = (ymin + ymax) / 2.0;
    float width = abs(xmax - xmin);
    float height = abs(ymax - ymin);

    float width_add = width * 0.25;  // add width on one side
    float crop_width = width + width_add * 2.0;
    float crop_xmin = xmin - width_add;
    float crop_xmax = xmax + width_add;

    float height_add = height * ratio;
    float crop_height = height + height_add * 2;
    float crop_ymin = centery - crop_height / 2.0;
    float crop_ymax = centery + crop_height / 2.0;
    if (crop_width > img_width) {
    //    cout << "hconcat started " << endl;
        crop_xmin = 0;
        crop_xmax = crop_width;
     /*   char cw[100], iw[100], ih[100];
        sprintf(cw, "%.3f", crop_width);
        sprintf(iw, "%.3f", img_width);
        sprintf(ih, "%.3f", img_height);
     //   cout << "crop_width " << String(cw) << endl;
    //    cout << "img width " << String(iw) << "img height " << String(ih);

        Mat cols = Mat::zeros(img_height, int(crop_width) - img_width + 1, img.type()); // +1 for input > 0
        cv::hconcat(img, cols, img);*/

    }

    else if (crop_xmin < 0) {
        crop_xmin = 0;
        crop_xmax = crop_width;
    }
    else if (crop_xmax >= img_width) {
        crop_xmax = img_width;
        crop_xmin = img_width - crop_width;
    }
    if (crop_height > img_height) {
        crop_ymin = 0;
        crop_ymin = crop_height;
    //    cout << "add rows started" << endl;

   //     Mat rows = Mat::zeros(int(crop_height) - img_height + 1, img.cols, img.type()); // +1 for input > 0
     //   img.push_back(rows);

    }

    else if (crop_ymin < 0) {
        crop_ymin = 0;
        crop_ymax = crop_height;
    }
    else if (crop_ymax >= img_height) {
        crop_ymax = img_height;
        crop_ymin = img_height - crop_width;
    }
    rectangle(tmp, Rect(crop_xmin, crop_ymin, crop_xmax-crop_xmin, crop_ymax-crop_ymin), Scalar(255,0,0)); //, 'red'
    *cymin = crop_ymin;
    *cymax = crop_ymax;
}

static vector<Rect> forbidden_area(float xmin, float ymin, float xmax, float ymax) {
    vector<Rect> fob;

    float width = xmax - xmin;
    float height = ymax - ymin;
    float centerx = (xmin + xmax) / 2;
    float centery = (ymin + ymax) / 2;
    // background : not in all
    fob.push_back(Rect(xmin, ymin, width, height));
    // nianjianbiao: not in center
    fob.push_back(Rect(centerx - width/2/2 , centery - height/3/2, width/2, height/3));
    // zheyangban: not in bottom
    fob.push_back(Rect(xmin, ymin + height*2/3, width, height/2));
    // qita: not in upper
    fob.push_back(Rect(xmin, ymin, width, height/3));
    // anquandai: not in center
    fob.push_back(Rect(centerx - width/10/2 , centery - height/2, width/10, height));
    // guazhui: not in left and right, only in right for programming simplicity
    fob.push_back(Rect(xmin, ymin, width/4, height));
//	Rect x6 = Rect(xmin + width/4*3, ymin, width/4, height);
    // zhijinhe: not in upper
    fob.push_back(Rect(xmin, ymin, width, height/2.5));

    return fob;
}

}

#endif /* SRC_ALG_CAFFE_HELPER_H_ */
