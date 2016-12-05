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
void load_names(const string &name_list, vector<string> &names) {
    ifstream fp(name_list);
    if (!fp)
    {
    	cout << "Can't open list file " << name_list << endl;
    	exit(-1);
    }
    names.resize(0);
    string name;
    while (getline(fp, name)) {
        names.push_back(name);
    }
}

Bbox max_box(vector<Bbox> boundingBox) {
	float area = 0;
	Bbox output_box;
	for(size_t box_id = 0; box_id < boundingBox.size(); ++box_id) {
		if (boundingBox[box_id].second.area() > area) {
			area = boundingBox[box_id].second.area();
			output_box.first = boundingBox[box_id].first;
			output_box.second = boundingBox[box_id].second;
		}
	}
	return output_box;
}

int main(int argc, char const *argv[])
{
	if (argc > 3 || argc == 1)
	{
		cout << "Number of argments not match." << endl;
		exit(-1);
	}
    string name_txt = argv[1];
	string fea_dir;
	if (argc == 3) {
		fea_dir = argv[2];
	}

	// FileConfig config("config.txt");
    // if (!config.Parse()) {
    //     cerr << "Failed to parse config file." << endl;
    //     exit(1);
    // }
    
    vector<string> names;
    load_names(name_txt, names);

    Detector  *detector 		= create_detector(det_method::FCN, "models/detetor_0.1.0", 0);
	Alignment *alignment 		= create_alignment(align_method::CDNN, "models/alignment_0.4.2/", -1);
	Transformation *transformation   = create_transformation(transform_method::CDNN, "");
	Recognition *recognition 	= create_recognition(recog_method::FUSION,"models/recognition_0.4.1",0,true );
	// Verification *verification 	= create_verifier(verif_method::EUCLID);

	vector<RecogResult> recognitions(names.size());
    ofstream not_det("not_det.log");
    // clock_t start, finish;
    // double duration = 0;

    for (size_t i = 0; i < names.size(); ++i)
    //for (size_t i = 0; i < 1; ++i)
    {
        
	    // Recognition *recognition = create_recognition();
	    Mat img = imread(names[i]);
	    if (img.empty())
	    {
	    	cerr << "can't read image" << names[i];
	    	exit(-1);
	    }
        cout << names[i] << endl;
	    
	    Mat img_draw = img.clone();

		vector<Mat> imgs {img};
		vector<DetectResult> detect_result;
	    detector->detect(imgs, detect_result);
        if(detect_result[0].boundingBox.size() == 0){
            not_det << names[i] << endl;
            continue;
        }

	   

        cout << "detect " << detect_result[0].boundingBox.size() << "faces" << endl;
		vector<RecogResult> one_img_recog(detect_result[0].boundingBox.size());
	    for(size_t det = 0; det < detect_result[0].boundingBox.size(); ++det) {


		    AlignResult align_result = {};
		    alignment->align(img, detect_result[0].boundingBox[det].second, align_result, false);
		    // alignment->align(img, detect_result[0].boundingBox[det].second, align_result, false);

			// Mat transformed_img;
			// AlignResult transformed_align_result = {};
			// transformation->transform(img, align_result, transformed_img, transformed_align_result);
			// Mat draw_transformed = transformed_img.clone();
			// drawLandmarks(draw_transformed, transformed_align_result);
			// imwrite("trans_test.png", draw_transformed);

		    // alignment->align(img, detect_result[0].boundingBox[det].second, align_result, false);
            // putText(img_draw, to_string(align_result.score), Point(face_crop.x, face_crop.y+20),  FONT_HERSHEY_SIMPLEX, 1, Scalar(0,0,255));
            float det_score = detect_result[0].boundingBox[det].first;
            float align_score = align_result.score;
            // putText(img_draw, to_string(det_score+align_score*10), Point(face_crop.x, face_crop.y+40),  FONT_HERSHEY_SIMPLEX, 1, Scalar(0,0,255));
            cout << "det_score: " << det_score << endl;
            cout << "align_score: " << align_score << endl;
            cout << "fuse_score: "  << det_score + align_score * 5 << endl;
            if(!alignment->is_face(det_score, align_score, 1.4)) {
                cout << "not face " << names[i]<< endl;
                continue;
            }

	        // Mat img_draw_landmarks = align_result.face_image.clone();
		    // draw_landmarks(img_draw_landmarks, align_result);
		    // string draw_name = "test_draw_" + to_string(i) + "_" + to_string(det) + ".png";
		    // imwrite(draw_name, img_draw_landmarks);
		    if(!valid_landmarks(align_result, img.size())) {
			cerr << "can't align image!" <<endl;
			continue;
		    }

		    vector<Mat> faces {img};
		    vector<AlignResult> alignments {align_result};
		    vector<RecogResult> recog_result;
            // start = clock();////////-------------->
		    recognition->recog(faces, alignments, recog_result, "NONE");
            // finish = clock();//////////<--------------------
            // duration += static_cast<double>(finish - start) / CLOCKS_PER_SEC;
		    
			cout << "feature size: " <<  recog_result[0].face_feat.size()<<endl;
			one_img_recog[det] = recog_result[0];
		    cout << "Recognized!" <<endl;
	    }

        int pos = names[i].rfind("/");
        string des = names[i].substr(pos + 1, names[i].length() - 4);
		string draw_name0 = des + "_det_test_draw_" + to_string(i) + ".png";
		// imwrite(draw_name0, img_draw);
		if(!fea_dir.empty()) {
			bool save_ret = saveFeature(fea_dir + des + ".fea", one_img_recog);
			if(save_ret == false) {
				cout << "can't save feature" << endl;
				return -1;
			}
		}
		
	    
    }
    not_det.close();

    
    
	return 0;
}
 
