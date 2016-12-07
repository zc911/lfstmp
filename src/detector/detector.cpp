#include <cassert>
#include "detector.h"
#include "dlib/image_processing.h"
#include "dlib/image_transforms.h"
#include "dlib/opencv.h"
#include "dlib/pixel.h"
#include "dlib_utils.h"

using namespace std;
using namespace cv;
namespace DGFace{

Detector::Detector(int img_scale_max, int img_scale_min, bool is_encrypt)
        : _img_scale_max(img_scale_max), 
		_img_scale_min(img_scale_min),
		_is_encrypt(is_encrypt){
    assert(img_scale_min < img_scale_max);
}

Detector::~Detector() {
}

// preprocess, add black edge to make batch processing aplicable for inputs with different image sizes
void Detector::edge_complete(vector<cv::Mat> &imgs)
{
    for(size_t i = 0; i < imgs.size(); i++)
    {
        Mat mask = Mat::zeros(_max_image_size,CV_8UC3);
        Rect img_rect(0, 0, imgs[i].cols, imgs[i].rows);
        imgs[i].copyTo(mask(img_rect));
        imgs[i] = mask;
    } 
}

void Detector::detect(const vector<Mat> &imgs, vector<DetectResult> &results) {
	detect_impl(imgs, results);
}

/*------------------------------------
void Detector::detect(const vector<Mat> &imgs, vector<DetectResult> &results) {
    vector<Mat> resized_imgs;
    resized_imgs.reserve(imgs.size());
    vector<float> scale_ratios;
    resized_imgs.resize(imgs.size());
    scale_ratios.resize(imgs.size());
    results.resize(imgs.size());
    _max_image_size = Size(0, 0);
    for (size_t idx = 0; idx < imgs.size(); idx++) {
        Mat img = imgs[idx];
        Mat resized_img;
        results[idx].image_size = Size(img.cols, img.rows); //record the original img size
        Size image_resize;
        // image resize, short edge is up to _img_scale
        float resize_ratio = 1;
        if (img.rows > _img_scale_max && img.cols > _img_scale_max) {
            resize_ratio = float(_img_scale_max) / min(img.cols, img.rows);
        } else if (img.rows < _img_scale_min || img.cols < _img_scale_min) {
            resize_ratio = float(_img_scale_min) / min(img.cols, img.rows);
        }
        //cout << img.cols * resize_ratio << "cols, rows" <<  img.rows * resize_ratio << endl;
        int width  = img.cols * resize_ratio;
        int height = img.rows * resize_ratio;

        // record the maximum width and height as mask
        _max_image_size.width  = max(_max_image_size.width, width);
        _max_image_size.height = max(_max_image_size.height, height);
        image_resize = Size(width, height);
        resize(img, resized_img, image_resize);
       
        resized_imgs[idx] = resized_img;
        scale_ratios[idx] = resize_ratio;
        // cout << "ratio = " << resize_ratio << "w = " << resized_imgs[idx].cols << "\th = " << resized_imgs[idx].rows << endl;
    }

    // Add black edge to support batch process for images with different sizes 
    edge_complete(resized_imgs);

    detect_impl(resized_imgs, results);
    for (size_t idx = 0; idx < imgs.size(); idx++) {
        Rect img_bbox = Rect(0, 0, imgs[idx].cols, imgs[idx].rows);
        auto &bboxes = results[idx].boundingBox;
        float ratio  = scale_ratios[idx];
        for (size_t i = 0; i < bboxes.size(); i++) {
            Rect &bbox   = bboxes[i].second;
            bbox.x      /= ratio;
            bbox.y      /= ratio;
            bbox.width  /= ratio;
            bbox.height /= ratio;
            bbox &= img_bbox;
            assert(bbox.x >= 0);
            assert(bbox.y >= 0);
            assert(bbox.x + bbox.width <= imgs[idx].cols);
            assert(bbox.y + bbox.height <= imgs[idx].rows);

            // Rect rect = results[idx].boundingBox[0].second;
            // cout << "bboxes: x=" << rect.x << ", y = " << rect.y << ", width = " << rect.width << ", height = " << rect.height << endl;
        }
    }
}
*/

}
