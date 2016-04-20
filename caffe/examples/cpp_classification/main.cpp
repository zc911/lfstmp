#include <iostream>
#include "faster_rcnn.h"

using namespace std;
using namespace cv;

void print_time(struct timeval a, struct timeval b, string task_name) {
    double timecost = (double(b.tv_sec - a.tv_sec) + double(b.tv_usec - a.tv_usec) / 1.0e6);
    cout << task_name << " timecost: " << timecost << endl;
}

void DetectionForCamera()
{
	cout<<"loading model..."<<endl;
	//string model_file   = "/home/zz/code/faster_rcnn/py-faster-rcnn/models/VGG_CNN_M_1024_face/faster_rcnn_end2end/test_c.prototxt";
	//string model_file   = "/mnt/data1/zdb/work/caffe_for_frcnn/zuozhen/py-faster-rcnn/models/GoogleNet_inception5/faster_rcnn_end2end/test.prototxt";
	//string model_file   = "../models/GoogleNet_inception5/faster_rcnn_end2end/test.prototxt";
	//string trained_file = "./googlenet_faster_rcnn_iter_350000.caffemodel";
	string model_file   = "../models/GoogleNet/faster_rcnn_end2end/deploy.prototxt";
	string trained_file = "./googlenet_faster_rcnn_iter_350000.caffemodel";
	// string trained_file = "/home/zz/code/faster_rcnn/py-faster-rcnn/output/faster_rcnn_end2end/FDDB_fold_train/vgg_cnn_m_1024_faster_rcnn_face_iter_70000.caffemodel";
    string layer_name_rois  = "rois";
    string layer_name_score = "cls_prob";
    string layer_name_bbox  = "bbox_pred";

    int scale = 400;
    float det_thresh = 0.5;
    int max_per_img = 100;
	Faster_rcnn detector(model_file, trained_file, layer_name_rois, layer_name_score, layer_name_bbox, true, 1, scale, det_thresh, max_per_img);
	//Faster_rcnn detector(model_file, trained_file, layer_name_rois, layer_name_score, layer_name_bbox, false, 1, scale, det_thresh, max_per_img);

    cout<<"start!"<<endl;
	int frame_count = 1;

    vector<Mat> images;
    vector<Blob<float>* > outputs;
    vector<struct Bbox> result;

    string filename;
	while (cin >> filename) {
        Mat frame = imread(filename.c_str(), -1);

        cout << frame.rows << " " << frame.cols << endl;

        images.resize(0);
		images.push_back(frame);

	    struct timeval start;
	    gettimeofday(&start, NULL);

        cout << "before feedard" << endl;
        detector.forward(images, outputs);
        cout << "feedard" << endl;

		struct timeval mid;
		gettimeofday(&mid, NULL);
        print_time(start, mid, "forward");

        detector.get_detection(outputs, result);

		struct timeval mid_get;
		gettimeofday(&mid_get, NULL);
        print_time(mid, mid_get, "get_detection");

        vector<Scalar> colors;
        colors.push_back(Scalar(0,0,0));
        colors.push_back(Scalar(255,0,0));
        colors.push_back(Scalar(0,255,0));
        colors.push_back(Scalar(0,0,255));
        colors.push_back(Scalar(255,255,0));

        vector<string> class_names;
        class_names.push_back("bg");
        class_names.push_back("car");
        class_names.push_back("person");
        class_names.push_back("bike");
        class_names.push_back("tricycle");
		for(size_t bbox_id = 0; bbox_id < result.size(); bbox_id ++) {
            int cls_id = result[bbox_id].cls_id;
            if ((cls_id == 1 && result[bbox_id].confidence > 0.8)
            || (cls_id == 2 && result[bbox_id].confidence > 0.5)
            || (cls_id == 3 && result[bbox_id].confidence > 0.5)
            || (cls_id == 4 && result[bbox_id].confidence > 0.5)
            ) {
			    rectangle(frame, result[bbox_id].rect, colors[cls_id],3);
			    char str_prob[100];
			    sprintf(str_prob,"%s_%.3f", class_names[cls_id].c_str(), result[bbox_id].confidence);
			    string info(str_prob);
			    putText(frame, info, Point(result[bbox_id].rect.x, result[bbox_id].rect.y), CV_FONT_HERSHEY_SIMPLEX, 0.5,  Scalar(0,0,255));
            }
		}

		struct timeval end;
		gettimeofday(&end, NULL);
		double time_per_frame = (double(end.tv_sec - start.tv_sec) + double(end.tv_usec - start.tv_usec) / 1.0e6);
		cout << "time_per_frame = "<< time_per_frame << endl;
		cv::imshow("Cam",frame);
        cv::waitKey(-1);
		//if (cv::waitKey(1)>=0)
	      //stop = true;

	}
}

int main(int argc, char** argv)
{
    google::InitGoogleLogging(argv[0]);
	DetectionForCamera();
	return 0;
}
