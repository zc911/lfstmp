
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

list<string> load_names(const string &name_list) {
    ifstream fp(name_list);
    if (!fp)
    {
    	cout << "Can't open list file " << name_list << endl;
    	exit(-1);
    }

    list<string> names;
    string name;
    while (getline(fp, name)) {
        names.push_back(move(name));
    }
	return names;
}

vector<string> split_name(const string& img_name) {
	auto sp_pos = img_name.find_last_of('/');
	string real_img_name;
	if(sp_pos == string::npos) {
		real_img_name = img_name;
	}else {
		real_img_name = img_name.substr(sp_pos + 1);
	}
	auto dash_pos = real_img_name.find_last_of('-');
	auto undersroce_pos = real_img_name.find_last_of('_');
	auto point_pos = real_img_name.find_last_of('.');

	vector<string> ret_str(3);
	ret_str[0] = real_img_name.substr(0, dash_pos);
	ret_str[1] = real_img_name.substr(dash_pos + 1, undersroce_pos - dash_pos -1);
	ret_str[2] = real_img_name.substr(undersroce_pos + 1, point_pos - undersroce_pos - 1);

	return ret_str;
}

bool is_daytime(const string& local_time) {
	int hour_time = stoi(local_time.substr(0,2)) + 8;
	hour_time = hour_time > 24 ? hour_time - 24 : hour_time;

	return (hour_time > 7 && hour_time < 18);
}

int main(int argc, char const *argv[]) {
	if (argc != 3) {
		cout << "Number of argments not match." << endl;
		exit(-1);
	}

	string name_txt = argv[1];
	string out_txt = argv[2];

	auto name_list = load_names(name_txt);
	cout << "total " << name_list.size() << " files" << endl;

    //Detector  *detector 		= create_detector(det_method::FCN, "models/detector_0.1.0", 0);
    Detector  *detector 		= create_detector(det_method::SSD, "models/detector_ssd", 0);
	//Alignment *alignment 		= create_alignment(align_method::CDNN, "models/alignment_0.4.2/", -1);

	ofstream err_log("err.log");
	ofstream mark_log(out_txt);
	for(auto& one_file: name_list) {
		auto file_infos = split_name(one_file);
		if(!is_daytime(file_infos[2])) {
			err_log << "not in daytime " << one_file << endl;
			continue;
		}

		vector<Mat> imgs(1); 
		imgs[0] = imread(one_file);
		
		if(imgs[0].empty()) {
			cout << "can't read image " << one_file << endl;
			err_log << "can't read image " << one_file << endl;
			continue;
		}

		vector<DetectResult> detect_result;
		detector->detect(imgs, detect_result);

        if(detect_result[0].boundingBox.size() == 0){
            err_log << "can't detect face " << one_file << endl;
			cout << "can't detect face " << one_file << endl;
            continue;
        }

		mark_log << one_file << " " << detect_result[0].boundingBox.size() << endl;
	}

	err_log.close();
	mark_log.close();
	return 0;
}
