#include <iostream>
#include <fstream>
#include <string>
#include <utility>
#include <algorithm>
#include <chrono>

#include <opencv2/opencv.hpp>

#include <detector.h>
#include "dgface_utils.h"
#include "dgface_config.h"
#include "alignment.h"
#include "recognition.h"


using namespace cv;
using namespace std;
using namespace DGFace;
struct image_info {
    string name;
    Mat image;
};

vector <string> load_names(const string &name_list) {
    ifstream fp(name_list);
    if (!fp) {
        cout << "Can't open list file " << name_list << endl;
        exit(-1);
    }

    vector <string> names;
    string name;
    while (getline(fp, name)) {
        names.push_back(std::move(name));
    }
    return names;
}
vector <image_info> read_images(const vector <string> &name_list) {
    vector <image_info> name_img_list;
    for (auto &one_name: name_list) {
        image_info img_tmp;
        img_tmp.name = one_name;
        img_tmp.image = imread(img_tmp.name);

        if (!img_tmp.image.empty()) {
            name_img_list.push_back(std::move(img_tmp));
        }
    }
    return name_img_list;
}
template<class T>
vector <vector<T>> split_list(const vector <T> &image_list, int sub_vec_len) {
    vector <vector<T>> splited_list;
    for (size_t i = 0; i < image_list.size(); i += sub_vec_len) {
        auto start_i = i;
        auto end_i = i + sub_vec_len;
        if (end_i < image_list.size()) {
            splited_list.push_back(std::move(vector<T>(image_list.begin() + start_i, image_list.begin() + end_i)));
        } else {
            end_i = image_list.size();
            splited_list.push_back(std::move(vector<T>(image_list.begin() + start_i, image_list.begin() + end_i)));
            break;
        }
    }
    return splited_list;
}
template<class T>
vector <T> merge_list(const vector <vector<T>> &t_list) {
    vector <T> merged_list;
    for (auto &one_sub_list : t_list) {
        merged_list.insert(merged_list.end(), one_sub_list.begin(), one_sub_list.end());
    }
    return merged_list;
}

Bbox max_box(vector <Bbox> boundingBox) {
    float area = 0;
    Bbox output_box;
    for (size_t box_id = 0; box_id < boundingBox.size(); ++box_id) {
        if (boundingBox[box_id].second.area() > area) {
            area = boundingBox[box_id].second.area();
            output_box.first = boundingBox[box_id].first;
            output_box.second = boundingBox[box_id].second;
        }
    }
    return output_box;
}

int main(int argc, char const *argv[]) {
	if (argc != 4) {
		cout << "Number of argments not match." << endl;
		exit(-1);
	}
    string name_txt = argv[1];

    bool visualize = static_cast<bool>(atoi(argv[2]));
    int batch_size = static_cast<int>(atoi(argv[3]));

    vector<string> names = load_names(name_txt);
    auto img_list = read_images(names);
    auto splited_list = split_list(img_list, batch_size);
    cout << img_list.size() << endl;
    cout << splited_list.size() << endl;

    Detector  *detector 		= create_detector(det_method::FCN, "../../data/model/detector/fcn/0.1.0", 0);
    // Detector  *detector 		= create_detector(det_method::SSD, "data/model/detector/ssd/0.0.3", 0);
    // Detector  *detector 		= create_detector(det_method::SSD, "data/model/detector/ssd/0.0.4", 0);
	// Alignment *alignment 		= create_alignment(align_method::CDNN, "models/alignment_0.4.2/", -1);
	// Transformation *transformation   = create_transformation(transform_method::CDNN, "");
	// Recognition *recognition 	= create_recognition(recog_method::FUSION,"models/recognition_0.4.1",0,true );
	// Verification *verification 	= create_verifier(verif_method::EUCLID);
	// Detector *detector = create_detector_with_global_dir(det_method::FCN, "../../data/", 0);

    //ofstream not_det("not_det.log");

    vector<DetectResult> detect_results;

    //chrono::duration<double> time_span(0.0);
    for(const auto& one_batch: splited_list) {

	vector<Mat> imgs(one_batch.size());
        transform(one_batch.begin(), one_batch.end(), imgs.begin(),
                    [](const image_info& one_info) {return one_info.image;});
	vector<DetectResult> curr_result;

        //auto time_start = chrono::high_resolution_clock::now();
	detector->detect(imgs, curr_result);
        //auto time_finish = chrono::high_resolution_clock::now();
        //time_span += chrono::duration_cast<chrono::duration<double> >(time_finish - time_start);
        detect_results.insert(detect_results.end(), curr_result.begin(), curr_result.end());
    }
    //cout << "compute time : " << time_span.count() << " seconds." << endl;

    assert(detect_results.size() == img_list.size());
    if(visualize) {
        for(size_t i = 0; i < img_list.size(); ++i) {
            int pos = img_list[i].name.rfind("/");
            string des = img_list[i].name.substr(pos + 1, img_list[i].name.length() - 4);
		    string draw_name0 = des + "_det_test_draw_" + to_string(i) + ".png";
	    	drawDetectionResult(img_list[i].image, detect_results[i], true);
            imwrite(draw_name0, img_list[i].image);
        }
    }

   ofstream det_result_file("det_info.txt");
   for(size_t i = 0; i < img_list.size(); ++i) {
   	det_result_file << img_list[i].name << endl;
   	const auto& all_det_bbox = detect_results[i].boundingBox;
   	det_result_file << all_det_bbox.size() << endl;
   	for(const auto& one_bbox: all_det_bbox) {
   		det_result_file << one_bbox.first;
   		auto common_box = one_bbox.second.boundingRect();
   		det_result_file << " " << common_box.x
				<< " " << common_box.y
				<< " " << common_box.width
				<< " " << common_box.height << endl;
   	}
   }

/*
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
			cout << names[i] << endl;
            continue;
        }
        cout << "detect " << detect_result[0].boundingBox.size() << "faces" << endl;

        int pos = names[i].rfind("/");
        string des = names[i].substr(pos + 1, names[i].length() - 4);
		string draw_name0 = des + "_det_test_draw_" + to_string(i) + ".png";
        if(visualize) {
	    	drawDetectionResult(img_draw, detect_result[0], true);
            imwrite(draw_name0, img_draw);
        }
	    
    }
    */
    delete detector;
    
	return 0;
//    not_det.close();
//    delete detector;

}
 
