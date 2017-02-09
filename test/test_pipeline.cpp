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
    int batch_size = 8;

    Detector  *detector 		= create_detector(det_method::FCN, "data/model/detector/fcn/0.1.0", 0);
    //Detector  *detector 		= create_detector(det_method::SSD, "models/detector_ssd", 0);
    Alignment *alignment 		= create_alignment(align_method::CDNN, "data/model/alignment/cdnn/0.4.2", -1);
	Transformation *transformation   = create_transformation(transform_method::CDNN, "");
	//Recognition *recognition 	= create_recognition(recog_method::CDNN_CAFFE,"data/model/recognition/cdnn_caffe/0.0.5", 0, true, false, batch_size);
	//Recognition *recognition 	= create_recognition(recog_method::CDNN_CAFFE,"data/model/recognition/cdnn_caffe/0.1.0", 0, true, false, batch_size);
	Recognition *recognition 	= create_recognition(recog_method::CDNN_CAFFE,"data/model/recognition/cdnn_caffe/max_pooling", 0, true, false, batch_size);
	//Recognition *recognition 	= create_recognition(recog_method::CNN,"data/model/recognition/LCNN/0.1.0", 0, true, false, batch_size);
	//Recognition *recognition 	= create_recognition(recog_method::FUSION,"models/recognition_0.4.1",0,true );
	Verification *verification 	= create_verifier(verif_method::EUCLID);

    ofstream not_det("not_det.log");
    // clock_t start, finish;
    // double duration = 0;

	vector<RecogResult> recog_results;
	vector<RecogResult> recog_batch_results;
	vector<AlignResult> alignments;
    vector<Mat> faces;
    vector<string> names_batch;
    for (size_t i = 0; i < names.size(); ++i)
    {
        
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
			cout << "Not detected face:" << names[i] << endl;
            continue;
        }

        cout << "detect " << detect_result[0].boundingBox.size() << "faces from: " << names[i] << endl;
	    for(size_t det = 0; det < 1/*detect_result[0].boundingBox.size()*/; ++det) {
		    AlignResult align_result = {};
		    alignment->align(img, detect_result[0].boundingBox[det].second, align_result, false);
            float det_score = detect_result[0].boundingBox[det].first;
            float align_score = align_result.score;
            cout << "det_score: " << det_score << endl;
            cout << "align_score: " << align_score << endl;
            cout << "fuse_score: "  << det_score + align_score * 5 << endl;
            if(!alignment->is_face(det_score, align_score, 1.4)) {
                cout << "not face " << names[i]<< endl;
                continue;
            }

		    drawLandmarks(img_draw, align_result);
            // string draw_name = "test_draw_" + to_string(i) + "_" + to_string(det) + ".png";
		    // imwrite(draw_name, img_draw_landmarks);
		    if(!valid_landmarks(align_result, img.size())) {
			cerr << "can't align image!" <<endl;
			continue;
		    }

		    faces.push_back(img);
            alignments.push_back(align_result);
            names_batch.push_back(names[i]);
	    }

        // batch recognition
        cout << "faces size: " << faces.size() << endl;
        if (faces.size() == batch_size) {
            assert(faces.size() == alignments.size());
            recog_batch_results.clear();
            recog_batch_results.reserve(batch_size);
            clock_t start = clock();////////-------------->
            recognition->recog(faces, alignments, recog_batch_results, "NONE");
            clock_t finish = clock();//////////<--------------------
            double duration = (double)(finish - start) / CLOCKS_PER_SEC;
            cout << "recog time: " << duration << endl;

            //cout << "feature size: " <<  recog_result[0].face_feat.size()<<endl;
            //cout << "Recognized!" <<endl;
            if(!fea_dir.empty()) {
                for (size_t idx = 0; idx < recog_batch_results.size(); idx++) {
                    string name = names_batch[idx];
                    cout << name << endl;
                    string des = name.substr(name.rfind("/") + 1, name.rfind(".jpg"));
                    if (!saveFeature(fea_dir + "/"+ des + ".fea", vector<RecogResult> (1, recog_batch_results[idx]))) {
                        cout << "can't save feature" << endl;
                        return -1;
                    }
                }
            }
            faces.clear();
            alignments.clear();
            names_batch.clear();
        }
    }
    if (faces.size() > 0 && faces.size() < batch_size) {
        assert(faces.size() == alignments.size());
        recog_batch_results.clear();
        recog_batch_results.reserve(faces.size());
        recognition->recog(faces, alignments, recog_batch_results, "NONE");
        cout << "face size: " << faces.size() << "alignments size" << alignments.size() << "recog result size: " << recog_batch_results.size() << endl;
        if(!fea_dir.empty()) {
            for (size_t idx = 0; idx < recog_batch_results.size(); idx++) {
                string name = names_batch[idx];
                cout << name << endl;
                string des = name.substr(name.rfind("/") + 1, name.rfind(".jpg"));
                if (!saveFeature(fea_dir + "/"+ des + ".fea", vector<RecogResult> (1, recog_batch_results[idx]))) {
                    cout << "can't save feature" << endl;
                    return -1;
                }
            }
        }
    }

    for (size_t i = 0; i < names.size(); ++i) {
    }
    not_det.close();
	return 0;
}
 
