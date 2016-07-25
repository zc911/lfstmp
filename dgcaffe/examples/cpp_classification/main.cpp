
#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <ctime>
#include <cassert>
#include <algorithm>

#include <ctype.h>
#include <sys/time.h>

using namespace caffe;
using namespace cv;
using namespace std;

class CaffeClassifier {
 public:
  CaffeClassifier(const string& model_file,
             const string& trained_file,
             const bool use_GPU,
             const int batch_size);

  vector<Blob<float>* > PredictBatch(vector<Mat> imgs, float a, float b, float c);
 private:
  shared_ptr<Net<float> > net_;
  cv::Size input_geometry_;
  int num_channels_;
  int batch_size_;
  cv::Mat mean_;
  bool useGPU_;
};

CaffeClassifier::CaffeClassifier(const string& model_file,
                       const string& trained_file,
                       const bool use_GPU,
                       const int batch_size) {
   if (use_GPU) {
       Caffe::set_mode(Caffe::GPU);
       Caffe::SetDevice(0);
       useGPU_ = true;
   }
   else {
       Caffe::set_mode(Caffe::CPU);
       useGPU_ = false;
   }

  /* Set batchsize */
  batch_size_ = batch_size;

  /* Load the network. */
  net_.reset(new Net<float>(model_file, TEST));
  net_->CopyTrainedLayersFrom(trained_file);

  CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
  //CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

  Blob<float>* input_layer = net_->input_blobs()[0];
  num_channels_ = input_layer->channels();
  CHECK(num_channels_ == 3 || num_channels_ == 1)
    << "Input layer should have 1 or 3 channels.";
  input_geometry_ = cv::Size(input_layer->width(), input_layer->height());
}

static bool PairCompare(const std::pair<float, int>& lhs,
                        const std::pair<float, int>& rhs) {
  return lhs.first > rhs.first;
}

//std::vector< float >  CaffeClassifier::PredictBatch(const vector< cv::Mat > imgs) {
vector<Blob<float>* > CaffeClassifier::PredictBatch(vector< cv::Mat > imgs, float a, float b, float c) {
  Blob<float>* input_layer = net_->input_blobs()[0];
  
  input_geometry_.height = imgs[0].rows;
  input_geometry_.width = imgs[0].cols;
  input_layer->Reshape(batch_size_, num_channels_,
                       input_geometry_.height,
                       input_geometry_.width);
  
  float* input_data = input_layer->mutable_cpu_data();
  int cnt = 0;
  for(int i = 0; i < imgs.size(); i++) {
    cv::Mat sample;
    cv::Mat img = imgs[i];

    if (img.channels() == 3 && num_channels_ == 1)
      cv::cvtColor(img, sample, CV_BGR2GRAY);
    else if (img.channels() == 4 && num_channels_ == 1)
      cv::cvtColor(img, sample, CV_BGRA2GRAY);
    else if (img.channels() == 4 && num_channels_ == 3)
      cv::cvtColor(img, sample, CV_BGRA2BGR);
    else if (img.channels() == 1 && num_channels_ == 3)
      cv::cvtColor(img, sample, CV_GRAY2BGR);
    else
      sample = img;

    if((sample.rows != input_geometry_.height) || (sample.cols != input_geometry_.width)) {
        cv::resize(sample, sample, Size(input_geometry_.width, input_geometry_.height));
    }

    float mean[3] = {104, 117, 123};
    for(int k = 0; k < sample.channels(); k++) {
        for(int i = 0; i < sample.rows; i++) {
            for(int j = 0; j < sample.cols; j++) {
               input_data[cnt] = (float(sample.at<uchar>(i,j*3+k))-mean[k]);
               cnt += 1;
            }
        }
    }
  }
  /* Forward dimension change to all layers. */
  net_->Reshape();
 
  struct timeval start;
  gettimeofday(&start, NULL);

  net_->ForwardPrefilled();

  if(useGPU_) {
    cudaDeviceSynchronize();
  }

  struct timeval end;
  gettimeofday(&end, NULL);
  cout << "pure model predict time cost: " << (1000000*(end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec)/1000 << endl;

  /* Copy the output layer to a std::vector */
  vector<Blob<float>* > outputs;

  cout << "net_outputs: "<< net_->num_outputs() << endl;
  for(int i = 0; i < net_->num_outputs(); i++) {
    Blob<float>* output_layer = net_->output_blobs()[i];
    outputs.push_back(output_layer);
  }
  return outputs;
}

int main(int argc, char** argv) {

  google::InitGoogleLogging(argv[0]);

  // caffe variables
  string model_file   = argv[1];
  string trained_file = argv[2]; 
  string image_list = argv[3];
  string output_list = argv[4];
  float global_confidence = atof(argv[5]);

  CaffeClassifier ssd_detector(model_file, trained_file, true, 1);

  FILE *fcin  = fopen(image_list.c_str(),"r");
  if(!fcin) {
    cout << "can not open filelist" << endl;
  }
  char image_filename[200];

  int tot_cnt = 0;
  FILE* fid = fopen(output_list.c_str(), "w");
  while(fscanf(fcin, "%s", image_filename)!=EOF) {
        
        tot_cnt += 1;
	    fprintf(fid, "%s ", image_filename);
        cout << "filename " << string(image_filename) << endl;
        vector<Mat> images;
        Mat image = imread(image_filename, -1);
        if (image.empty()) {
            cout << "Wrong Image" << endl;
            continue;
        }
        Mat img;
        float border_ratio = 0.00;
        img = image.clone();

        int target_row = 400; //img.rows * enlarge_ratio;
        int target_col = 600; //img.cols * enlarge_ratio;

        float ratio_row = img.rows * 1.0 / target_row;
        float ratio_col = img.cols * 1.0 / target_col;

        resize(img, img, Size(target_col, target_row));

        images.push_back(img);

        //struct timeval end_1;
        //gettimeofday(&end_1, NULL);
        //cout << "resize image time cost: " << (1000000*(end_1.tv_sec - end_0.tv_sec) + end_1.tv_usec - end_0.tv_usec)/1000 << endl;

        struct timeval start;
        gettimeofday(&start, NULL);
        vector<Blob<float>* > outputs = ssd_detector.PredictBatch(images, 128.0, 128.0, 128.0);

        int box_num = outputs[0]->height();
        int box_length = outputs[0]->width();
        cout << outputs[0]->num() << " " << outputs[0]->channels() << " " << outputs[0]->height() << " " << outputs[0]->width() << endl;
        const float* top_data = outputs[0]->cpu_data();
        vector<Scalar> color;
        color.push_back(Scalar(255,0,0));
        color.push_back(Scalar(0,255,0));
        color.push_back(Scalar(0,0,255));
        color.push_back(Scalar(255,255,0));
        color.push_back(Scalar(0,255,255));
        color.push_back(Scalar(255,0,255));
        vector<string> tags;
        tags.push_back("bg");
        tags.push_back("car");
        tags.push_back("person");
        tags.push_back("bicycle");
        tags.push_back("tricycle");
        //tags.push_back("unknown");
        img = image;


        vector<float> conf;
        conf.push_back(-1);
        conf.push_back(0.9);
        conf.push_back(0.6);
        conf.push_back(0.6);
        conf.push_back(0.6);
        for(int j = 0; j < box_num; j++) {
            int cls = top_data[j * 7 + 1];
            float score = top_data[j * 7 + 2];
            float xmin = top_data[j * 7 + 3] * img.cols; 
            float ymin = top_data[j * 7 + 4] * img.rows; 
            float xmax = top_data[j * 7 + 5] * img.cols; 
            float ymax = top_data[j * 7 + 6] * img.rows; 
            //if (score > global_confidence[cls]) {
            if (score > conf[cls]) {
                char char_score[100];
                sprintf(char_score, "%.3f", score);
                rectangle(img, Rect(xmin, ymin, xmax-xmin, ymax-ymin), color[cls]);
                putText(img, tags[cls] + "_" + string(char_score), Point(xmin, ymin), CV_FONT_HERSHEY_COMPLEX, 0.5, color[0]);
            }
        }


        struct timeval end;
        gettimeofday(&end, NULL);
        cout << "total time cost: " << (1000000*(end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec)/1000 << endl;
        fprintf(fid, "\n");

        //cout << "rows: " << image.rows << " cols: " << image.cols << endl;
        //imwrite("debug.jpg", image);
        //char save_path[100]; 
        //sprintf(save_path, "debug/%d.jpg", tot_cnt);
        //imwrite(save_path, img);
        imshow("debug.jpg", img);
        waitKey(-1);
  }
  fclose(fid);
  return 0;
}
