#include <alignment.h>
#include "dlib/image_processing.h"
#include "dlib/image_transforms.h"
#include "dlib/opencv.h"
#include "dlib/pixel.h"
#include "dlib_utils.h"
#include "dgface_utils.h"

using namespace std;
using namespace cv;
namespace DGFace{

Alignment::Alignment(vector<int> face_size) : _face_size(face_size) {
    assert(_face_size.size() == 1 || _face_size.size() == 2);
    if (_face_size.size() == 1) {
        _face_size.push_back(_face_size[0]);
    } 
}

Alignment::~Alignment(void) {
}


void Alignment::align(const Mat &img_in, const RotatedRect& rot_bbox, AlignResult &result, bool adjust) {
	Mat img;
    if (img_in.channels() == 4)
        cvtColor(img_in, img, COLOR_BGRA2BGR);
    else if (img_in.channels() == 1)
        cvtColor(img_in, img, COLOR_GRAY2BGR);
    else img = img_in;

    align_impl(img, rot_bbox, result);
}

void Alignment::align(const Mat &img_in, const Rect &bbox, AlignResult &result, bool adjust) {
    Mat img;
    if (img_in.channels() == 4)
        cvtColor(img_in, img, COLOR_BGRA2BGR);
    else if (img_in.channels() == 1)
        cvtColor(img_in, img, COLOR_GRAY2BGR);
    else img = img_in;

    align_impl(img, bbox, result);
    if (!result.landmarks.size())
        return;

    Rect image_bbox = Rect(Point(0, 0), img.size());
    Rect rect = boundingRect(result.landmarks);
    rect &= image_bbox;

    if (adjust) {
        dlib::cv_image<dlib::bgr_pixel> dlib_img(img);
        // height: _face_size[0], width: _face_size[1]
        dlib::array2d<dlib::bgr_pixel> out(_face_size[0], _face_size[1]);
        vector<dlib::point> avg_points, face_points;
        cv_point2dlib(_avg_points, avg_points);
        dlib::point_transform_affine trans;
		vector<Point> dlib_landmarks;
        cv_point2dlib(dlib_landmarks, face_points);
		cvtPoint2iToPoint2f(dlib_landmarks, result.landmarks);
        //trans = dlib::find_affine_transform(avg_points, face_points);
        trans = dlib::find_similarity_transform(avg_points, face_points);
        dlib::transform_image(dlib_img, out, dlib::interpolate_bilinear(), trans);
        result.face_image = dlib::toMat(out).clone();
    } else {
        // modify the rect
        //int center_x = rect.x + rect.width / 2;
        //int center_y = rect.y + rect.height / 2;
        //if (rect.width > rect.height) {
        //    rect.y = center_y - rect.width / 2;
        //    rect.height = rect.width;
        //} else {
        //    rect.x = center_x - rect.height / 2;
        //    rect.width = rect.height;
        //}

        //if (rect.y < 0 )
        //{
        //    rect.y = 0;
        //}
        //else if (rect.y + rect.height > img.rows)
        //{
        //    rect.y = img.rows - rect.height;
        //}
        //if (rect.x < 0 )
        //{
        //    rect.x = 0;
        //}
        //else if (rect.x + rect.width > img.cols)
        //{
        //    rect.x = img.cols - rect.width;
        //}
        //result.face_image = img(rect); // .clone();
       // resize(img(bbox), result.face_image, Size(_face_size[1], _face_size[0]));
    }
    // imshow("face", result.face_image);
    // waitKey(500);
    result.bbox = rect; //update the bbox`
}

void Alignment::set_avgface(const Mat &img, const Rect &bbox) {
    Mat resized_img;
    AlignResult result;
    resize(img, resized_img, Size(_face_size[1], _face_size[0]));
    align_impl(resized_img, bbox, result);
    assert(result.landmarks.size());
	cvtPoint2fToPoint2i(result.landmarks, _avg_points);
    // _avg_points = result.landmarks;
}

bool Alignment::is_face(float det_score, float landmark_score, float det_thresh) {

    if(landmark_score >= 0.5){
        return true;
    }
    else if(det_score < 0.7){
        return false;
    }else{
        float fuse_score = det_score + landmark_score * 5;

        if(fuse_score < 1.4 && landmark_score < 0.0001){
                return false;
        } else{
            return(fuse_score >= det_thresh);
        }
    }
}
}
