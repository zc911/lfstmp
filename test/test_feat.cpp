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

int main(int argc, char const *argv[])
{
    string img_path = "/home/Dataset/code_new/FaceVeri/DGFaceSDK/faces/000057.jpg";

    int batch_size = 1;

    Recognition *recognition 	= create_recognition(recog_method::CNN,"data/model/recognition/LCNN/0.1.0", 0, true, false, batch_size);
#ifdef DEBUG
    ofstream flog("feat.log");
#endif
    vector<Mat> faces;
    Mat img = imread(img_path);
    faces.push_back(img);
    vector<AlignResult> alignments;
    alignments.resize(1);
    vector<RecogResult> recog_results;
    recognition->recog(faces, alignments, recog_results, "NONE");
    for (int i = 0; i < recog_results[0].face_feat.size(); i++) {
        flog << recog_results[0].face_feat[i] << " ";
    }
    flog << endl;
    flog.close();
	return 0;
}
 
