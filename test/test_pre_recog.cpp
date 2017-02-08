#include <sys/time.h>
#include "time.h"
#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/opencv.hpp>
#include <detector.h>
#include <alignment.h>
#include <recognition.h>
#include <verification.h>
#include <database.h>
#include <transformation.h>
#include "dgface_utils.h"
#include "dgface_config.h"


using namespace cv;
using namespace std;
using namespace DGFace;
//#define DEBUG true

bool valid_landmarks(const AlignResult &align_result, const Size& img_size) {
    auto &landmarks = align_result.landmarks;
    for (auto pt = landmarks.begin(); pt != landmarks.end(); ++pt)
    {
        // circle(img, *pt, 2, Scalar(0,255,0), -1);

        if (!pt->inside(Rect(0,0,img_size.width,img_size.height)))
        {
            return false;
        }
    }
    return true;
}


void Batch_det_align (vector<Mat>& imgs, vector<string>& names, Detector* detector, Alignment* alignment, Transformation* transformation, ofstream &transformed_lmk_file, ofstream &not_det_align) {
    vector<DetectResult> detect_result;
    detect_result.resize(0);
    detector->detect(imgs, detect_result);
    for (size_t img_idx = 0; img_idx < detect_result.size(); img_idx++) {
        size_t num_bbox = detect_result[img_idx].boundingBox.size();
        string name = names[img_idx];
        name = name.substr(0, name.find(".jpg"));
        if(num_bbox == 0){
            not_det_align << "det no face" << name << endl;
            cout << "det no face: " << name << endl;
            continue;
        }
        for (size_t bbox_idx = 0; bbox_idx < num_bbox; bbox_idx++) {
            char tmp[1024];
            sprintf(tmp, "%s_%lu.jpg", name.c_str(), bbox_idx);
            string new_img_path(tmp);
            AlignResult align_result = {};
            alignment->align(imgs[img_idx], detect_result[img_idx].boundingBox[bbox_idx].second, align_result, false);
            float det_score = detect_result[img_idx].boundingBox[bbox_idx].first;
            float align_score = align_result.score;
            //cout << "det_score: " << det_score << endl;
            //cout << "align_score: " << align_score << endl;
            //cout << "fuse_score: "  << det_score + align_score * 5 << endl;
            if(!alignment->is_face(det_score, align_score, 1.4)) {
                not_det_align << "align no face" << new_img_path << endl;
                cout << "align no face: " << new_img_path << endl;
                continue;
            }
            // string draw_name = "test_draw_" + to_string(i) + "_" + to_string(det) + ".png";
            // imwrite(draw_name, img_draw_landmarks);
            if(!valid_landmarks(align_result, imgs[img_idx].size())) {
                not_det_align << "is not face" << new_img_path << endl;
                cout << "not image!" << new_img_path <<endl;
                continue;
            }

#ifdef DEBUG
            cout << "new name: " << new_img_path << endl;
            Mat img_draw = imgs[img_idx].clone();
            drawLandmarks(img_draw, align_result);
            imshow("landmarks", img_draw);
            waitKey(0);
#endif

            // similarity transform
            AlignResult transformed_alignment = {};
            Mat transformed_img;
            transformation->transform(imgs[img_idx], align_result, transformed_img, transformed_alignment);
            vector<double> transformedLandmarks;
            cvtLandmarks(transformed_alignment.landmarks, transformedLandmarks);
            // save transformed face
            imwrite(new_img_path, transformed_img);
            // save lmks
            transformed_lmk_file << new_img_path;
            for (size_t lmk_id = 0; lmk_id < transformedLandmarks.size(); lmk_id++) {
                transformed_lmk_file << " " << transformedLandmarks[lmk_id];
            }
            transformed_lmk_file << endl;

        }
    }
    imgs.clear();
}

int main(int argc, char const *argv[])
{
    if (argc != 3)
    {
        cout << "Number of argments not match." << endl;
        exit(-1);
    }
    ifstream name_txt(argv[1]);
    ofstream transformed_lmk_file(argv[2]);
    ofstream not_det_align("not_det_or_align.log");

    int batch_size = 10;
    Detector  *detector 		= create_detector(det_method::FCN, "data/model/detector/fcn/0.1.0", 0, false, batch_size);
    //Detector  *detector 		= create_detector(det_method::SSD, "models/detector_ssd", 0);
    Alignment *alignment 		= create_alignment(align_method::CDNN, "data/model/alignment/cdnn/0.4.2", -1);
    Transformation *transformation   = create_transformation(transform_method::CDNN, "");

    // clock_t start, finish;
    // double duration = 0;

    vector<Mat> imgs;
    vector<string> names;
    string line;
    string tar_path;
    string pre_tar_path = "";
    string img_path;
    string fold_path;
    int processed_tars = 0;
    int num_imgs = 0;
    while (getline(name_txt, line)) {
        size_t split_point = line.find(' ');
        tar_path = line.substr(0, split_point);
        img_path = "/home/zz/temp/" + line.substr(split_point+1);
        if (!tar_path.empty() && tar_path != pre_tar_path) {
            //if (!pre_tar_path.empty()) {
                system("sudo umount /home/zz/temp");
            //} 
            system(("archivemount -o readonly " + tar_path + " /home/zz/temp").c_str());
            pre_tar_path = tar_path;
            string tmp = tar_path.substr(28);
            tmp = tmp.substr(0, tmp.find(".tar"));
            fold_path = "/home/zz/Dataset/RR_neg/" + tmp;
            system(("mkdir -p " + fold_path).c_str());
#ifdef DEBUG
            cout << "tar_path: " << tar_path << ", img_path: " << img_path << endl;
            cout << "archivemount -o readonly " + tar_path + " /home/zz/temp" << endl;
            cout << "tar id: " << processed_tars++ << endl;
#endif
        }

        Mat img = imread(img_path);
        if (img.empty())
        {
            cerr << "can't read image: " << img_path;
            exit(-1);
        }
        imgs.push_back(img);
        string name = fold_path + img_path.substr(img_path.rfind('/'));
#ifdef DEBUG
        cout << "fold path: " << fold_path << ", new path: " << name << endl;
#endif
        names.push_back(name);
        if (imgs.size() == batch_size) {
            Batch_det_align(imgs, names, detector, alignment, transformation, transformed_lmk_file, not_det_align);
            imgs.clear();
            names.clear();
        } 
        num_imgs++;
        if (num_imgs % 100 == 0) {
            cout << "============================" << endl;
            cout << "processed imgs: " << num_imgs << endl;
        }
    }

    if (imgs.size() > 0 && imgs.size() < batch_size) {
        Batch_det_align(imgs, names, detector, alignment, transformation, transformed_lmk_file, not_det_align);
    }
    cout << "Total processed imgs: " << num_imgs << endl;
    name_txt.close();
    transformed_lmk_file.close();
    not_det_align.close();
    return 0;
}

