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
#include "dgface_utils.h"
#include "dgface_config.h"


using namespace cv;
using namespace std;
using namespace DGFace;

#define DEBUG true;
//#define FUSE true;

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

void extract_name(const vector<string> &full_path, vector<string>& result) {
		result.resize(full_path.size());
		for (size_t i = 0; i < full_path.size(); i++) {
				string item;
				string path = full_path[i];
				size_t index_slash = path.rfind('/');
				size_t index_dot   = path.rfind('.');
				string name = path.substr(index_slash + 1, index_dot - index_slash - 1);
				string::iterator iter = name.begin();
				while (iter != name.end()) {
						char ch = *iter++;
						if (ch < 0) {
								item += ch;
						}
				}
				result[i] = item;
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

vector<vector<float> > compute_scores(const vector<RecogResult>& recog_features, const int num_gallary, Verification* verification) {
    int num_query = recog_features.size() - num_gallary;
	vector<vector<float> > scores;
	scores.reserve(num_query);

	for(size_t idx = num_gallary; idx < recog_features.size(); idx++) {
        auto& query_feature = recog_features[idx];
		vector<float> curr_score;
		
		if(query_feature.face_feat.size() != 0) {
			curr_score.resize(num_gallary, 0.0f);
			for(size_t i = 0; i < num_gallary; i++) {
				auto& gallary_feature = recog_features[i];
				if(gallary_feature.face_feat.size() != 0) {
					curr_score[i] = verification->verify(query_feature.face_feat, gallary_feature.face_feat);
				}
			}
		}

		scores.push_back(move(curr_score));
	}
	return scores;
}

vector<vector<bool> > get_labels(const vector<string>& names, const int num_gallary) {
    int num_query = names.size() - num_gallary;
    vector<vector<bool> > labels;
    labels.reserve(num_query);

    for(size_t idx = num_gallary; idx < names.size(); idx++) {
        auto& query_name = names[idx];
        vector<bool> curr_label(num_gallary, false);
        for(size_t i = 0; i < num_gallary; i++) {
            auto pos = names[i].find(query_name);
            if(pos != string::npos) {
                curr_label[i] = true;
            }
        }
        labels.push_back(move(curr_label));
    }   
    return labels;
}



int main(int argc, char const *argv[])
{
	if (argc > 5 || argc == 1)
	{
		cout << "Number of argments not match." << endl;
		exit(-1);
	}
    string name_txt = argv[1];
    string gallary_txt = argv[2];
    string query_txt = argv[3];
	string fea_dir;
    if (argc == 5) {
        fea_dir = argv[4];
    }

	// FileConfig config("config.txt");
    // if (!config.Parse()) {
    //     cerr << "Failed to parse config file." << endl;
    //     exit(1);
    // }
    
    int batch_size = 2;//10;

    Detector  *detector 		= create_detector(det_method::FCN, "data/model/detector/fcn/0.1.0", 0);
    //Detector  *detector 		= create_detector(det_method::SSD, "data/model/detector/ssd", 0);
    Alignment *alignment 		= create_alignment(align_method::CDNN, "data/model/alignment/cdnn/0.4.2", -1);
	//Recognition *recognition 	= create_recognition(recog_method::CDNN_CAFFE,"data/model/recognition/cdnn_caffe/sigmoid", 0, true, false, batch_size);
	Recognition *recognition 	= create_recognition(recog_method::CDNN_CAFFE,"data/model/recognition/cdnn_caffe/0.1.0", 0, true, false, batch_size);
	//Recognition *recognition_step2 	= create_recognition(recog_method::CDNN_CAFFE,"data/model/recognition/cdnn_caffe/0.1.0", 0, true, false, batch_size);
	//Recognition *recognition 	= create_recognition(recog_method::CDNN_CAFFE,"data/model/recognition/cdnn_caffe/0.0.5", 0, true, false, batch_size);
	//Recognition *recognition = create_recognition(recog_method::CDNN_CAFFE,"data/model/recognition/cdnn_caffe/0.0.6.v2", 0, true, false, batch_size);
	//Recognition *recognition_step2 = create_recognition(recog_method::CDNN_CAFFE,"data/model/recognition/cdnn_caffe/posid_sample_apn", 0, true, false, batch_size);
	//Recognition *recognition = create_recognition(recog_method::CDNN_CAFFE,"data/model/recognition/cdnn_caffe/max_pooling", 0, true, false, batch_size);
	//Recognition *recognition = create_recognition(recog_method::CNN,"data/model/recognition/LCNN/0.0.0", 0, true, false, batch_size);
	//Recognition *recognition 	= create_recognition(recog_method::FUSION,"models/recognition_0.4.1",0,true );
	//Verification *verification 	= create_verifier(verif_method::EUCLID);
	Verification *verification 	= create_verifier(verif_method::COS);
	//Verification *verification 	= create_verifier(verif_method::NEUCLID);

#ifdef DEBUG
    ofstream not_det_align("not_det_or_align.log");
    ofstream not_face("not_face.log"); 
    ofstream low_score("recog_low_score.log");
    ofstream top1_score("top1_score.log");
#endif
    // clock_t start, finish;
    // double duration = 0;
    
    /////////////////////////////////////////////////////////////////////
    vector<string> gallary_path_list, gallary_name_list, query_path_list, query_name_list; 
    load_names(gallary_txt, gallary_path_list);
    extract_name(gallary_path_list, gallary_name_list);
    load_names(query_txt, query_path_list);
    extract_name(query_path_list, query_name_list);

    int num_gallary = gallary_path_list.size();
    int num_query = query_path_list.size();
    int num_g_new = num_gallary, num_q_new = num_query;
    vector<string> paths;
    paths = gallary_path_list;
    paths.insert(paths.end(), query_path_list.begin(), query_path_list.end());
    cout << "num_gallary: " << num_gallary << "num_query: " << num_query << endl;
    waitKey(0);

    vector<RecogResult> recog_results;
    vector<RecogResult> recog_results_step2;
	vector<RecogResult> recog_batch_results;
	vector<RecogResult> recog_batch_results_step2;
	vector<AlignResult> alignments;
    vector<Mat> faces;
    vector<string> paths_batch;
    vector<string> names;
    vector<string> paths_new;
    for (size_t i = 0; i < paths.size(); ++i)
    {
        
	    Mat img = imread(paths[i]);
	    if (img.empty())
	    {
	    	cerr << "can't read image" << paths[i];
	    	exit(-1);
	    }
        cout << paths[i] << endl;
	    
	    Mat img_draw = img.clone();

		vector<Mat> imgs {img};
		vector<DetectResult> detect_result;
	    detector->detect(imgs, detect_result);
        if(detect_result[0].boundingBox.size() == 0){
#ifdef DEBUG
            not_det_align << "det no face" << paths[i] << endl;
#endif
			cout << paths[i] << endl;
            if (i < num_gallary) {
                num_g_new--;
            } else {
                num_q_new--;
            }
            continue;
        }

        cout << "detect " << detect_result[0].boundingBox.size() << "faces from: " << paths[i] << endl;
        bool is_face = true;
	    for(size_t det = 0; det < 1/*detect_result[0].boundingBox.size()*/; ++det) {
		    AlignResult align_result = {};
		    alignment->align(img, detect_result[0].boundingBox[det].second, align_result, false);
            float det_score = detect_result[0].boundingBox[det].first;
            float align_score = align_result.score;
            cout << "det_score: " << det_score << endl;
            cout << "align_score: " << align_score << endl;
            cout << "fuse_score: "  << det_score + align_score * 5 << endl;
            if(!alignment->is_face(det_score, align_score, 1.4) || !valid_landmarks(align_result, img.size())) {
#ifdef DEBUG
                not_det_align << "align not face" << paths[i] << endl;
                not_face << paths[i] << endl;
#endif
                cout << "not face " << paths[i]<< endl;
                if (i < num_gallary) {
                    num_g_new--;
                } else {
                    num_q_new--;
                }
                is_face = false;
                break;
            }

            drawLandmarks(img_draw, align_result);
            // string draw_name = "test_draw_" + to_string(i) + "_" + to_string(det) + ".png";
		    // imwrite(draw_name, img_draw_landmarks);

		    faces.push_back(img);
            alignments.push_back(align_result);
            paths_batch.push_back(paths[i]);
        }

        cout << "is_face: " << is_face << "path idx: " << i << endl;
        if (!is_face) continue;
        if (i < num_gallary) {
            names.push_back(gallary_name_list[i]);
            paths_new.push_back(gallary_path_list[i]);
        } else {
            names.push_back(query_name_list[i - num_gallary]);
            paths_new.push_back(query_path_list[i - num_gallary]);
        }

        // batch recognition
        //cout << "faces size: " << faces.size() << endl;
        if (faces.size() == batch_size) {
            assert(faces.size() == alignments.size());
            recog_batch_results.clear();
            recog_batch_results.reserve(batch_size);
            // start = clock();////////-------------->
            recognition->recog(faces, alignments, recog_batch_results, "NONE");
            recog_results.insert(recog_results.end(), recog_batch_results.begin(), recog_batch_results.end());
#ifdef FUSE
            recog_batch_results_step2.clear();
            recog_batch_results_step2.reserve(batch_size);
            recognition_step2->recog(faces, alignments, recog_batch_results_step2, "NONE");
            recog_results_step2.insert(recog_results_step2.end(), recog_batch_results_step2.begin(), recog_batch_results_step2.end());
#endif
            // finish = clock();//////////<--------------------
            // duration += static_cast<double>(finish - start) / CLOCKS_PER_SEC;

            //cout << "feature size: " <<  recog_result[0].face_feat.size()<<endl;
            //cout << "Recognized!" <<endl;
            if(!fea_dir.empty()) {
                for (size_t idx = 0; idx < recog_batch_results.size(); idx++) {
                    string name = paths_batch[idx];
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
            paths_batch.clear();
        }
    }
    cout << "num_g_new: " << num_g_new << "num_q_new: " << num_q_new<< endl;

    if (faces.size() > 0 && faces.size() < batch_size) {
        assert(faces.size() == alignments.size());
        recog_batch_results.clear();
        recog_batch_results.reserve(faces.size());
        recognition->recog(faces, alignments, recog_batch_results, "NONE");
        recog_results.insert(recog_results.end(), recog_batch_results.begin(), recog_batch_results.end());
#ifdef FUSE
        recog_batch_results_step2.clear();
        recog_batch_results_step2.reserve(batch_size);
        recognition_step2->recog(faces, alignments, recog_batch_results_step2, "NONE");
        recog_results_step2.insert(recog_results_step2.end(), recog_batch_results_step2.begin(), recog_batch_results_step2.end());
#endif

        cout << "face size: " << faces.size() << "alignments size" << alignments.size() << "recog result size: " << recog_batch_results.size() << endl;
        if(!fea_dir.empty()) {
            for (size_t idx = 0; idx < recog_batch_results.size(); idx++) {
                string name = paths_batch[idx];
                cout << name << endl;
                string des = name.substr(name.rfind("/") + 1, name.rfind(".jpg"));
                if (!saveFeature(fea_dir + "/"+ des + ".fea", vector<RecogResult> (1, recog_batch_results[idx]))) {
                    cout << "can't save feature" << endl;
                    return -1;
                }
            }
        }
    }

    
	/////////////////////////////////////////////////////////////////////
    vector<vector<float> >scores = compute_scores(recog_results, num_g_new, verification);
#ifdef FUSE
    // feat wise combine
    // vector<RecogResult> recog_results_comb;
    // recog_results_comb.reserve(recog_results.size());
    // for (size_t idx = 0; idx < recog_results.size(); idx++) {
    //     RecogResult recog_feat_tmp = recog_results[idx];
    //     recog_feat_tmp.face_feat.insert(recog_feat_tmp.face_feat.end(), recog_results_step2[idx].face_feat.begin(), recog_results_step2[idx].face_feat.end());
    //     recog_results_comb.push_back(recog_feat_tmp);
    // }
    //
    // vector<vector<float> >scores = compute_scores(recog_results_comb, num_g_new, verification);
    // assert(scores.size() == labels.size());
    //
	/////////////////////////////////////////////////////////////////////
    // score combine
    // vector<vector<float> >scores = compute_scores(recog_results, num_g_new, verification);
    // vector<vector<float> >scores_step2 = compute_scores(recog_results_step2, num_g_new, verification);
    // assert(scores.size() == labels.size());
    // assert(scores_step2.size() == labels.size());
    // for (size_t idx = 0; idx < scores.size(); idx++) {
    //     for (size_t i = 0; i < scores[idx].size(); i++) {
    //         scores[idx][i] = (scores[idx][i] + scores_step2[idx][i] ) / 2.0;
    //     }
    // }
	/////////////////////////////////////////////////////////////////////
    // hierachical sort
    int step1_num_thresh = 10; //2200; //10;
    float score_thresh = .0f;
    vector<vector<float> >scores_step2 = compute_scores(recog_results_step2, num_g_new, verification);
    vector<vector<float> > top1_scores;
    for(size_t idx = 0; idx < scores.size(); idx++) {
        vector<float> curr_score = scores[idx];
        sort(curr_score.begin(), curr_score.end(), greater<float>());
        if (curr_score.size() > step1_num_thresh) {
            score_thresh = curr_score[step1_num_thresh];
        }
        float step2_top1 = 0;
        for (size_t i = 0; i < scores[idx].size(); i++) {
            if (scores[idx][i] < score_thresh) {
                scores[idx][i] = 0;
            } else {
                scores[idx][i] = scores_step2[idx][i];
                step2_top1 = max(step2_top1, scores[idx][i]);
            }
        }
        vector<float> tmp_top1_scores = {curr_score[0], step2_top1};
        top1_scores.push_back(tmp_top1_scores);
    }
#endif
     
    
	/////////////////////////////////////////////////////////////////////
    vector<vector<bool> >labels = get_labels(names, num_g_new);
    cout << "num of recog_results: " << recog_results.size() << "num of names: " << names.size() << endl;
    cout << "num of scores: " << scores.size() << "num of labels: " << labels.size() << "num of recog features: " << recog_results.size() << endl;
	vector<float> ap_vec;
	float mean_ap = computeMAP(scores, labels, ap_vec);
	for(size_t i = 0; i < ap_vec.size(); ++i) {
#ifdef DEBUG
#ifdef FUSE
		cout << paths_new[i+num_g_new] << " : " << ap_vec[i] << "\t" << "step1: " << top1_scores[i][0] << "\tstep2: " << top1_scores[i][1]<< endl;
        if (ap_vec[i] < 0.5) {
            low_score << paths_new[i+num_g_new] << endl;
            top1_score << paths_new[i+num_g_new] << "\t" << ap_vec[i] << "\t";
        } else {
            top1_score << paths_new[i+num_g_new] << "\t" << ap_vec[i] << "\t";
        }
        top1_score << top1_scores[i][0] << "\t";
        top1_score << top1_scores[i][1] << endl; 
#endif
#ifndef FUSE
		cout << paths_new[i+num_g_new] << " : " << ap_vec[i] << "\t" << scores[i][0] << endl;
#endif
#endif
	}
	cout << "mean-AP: " << mean_ap << endl;

#ifdef DEBUG
    not_det_align.close();
    not_face.close();
    low_score.close();
    top1_score.close();
#endif
	return 0;
}
 
