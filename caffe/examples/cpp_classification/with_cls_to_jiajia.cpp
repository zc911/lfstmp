
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

using namespace caffe;  // NOLINT(build/namespaces)
using namespace cv;
using namespace std;

/* Pair (label, confidence) representing a prediction. */
typedef std::pair<int, float> Prediction;

class CaffeClassifier {
 public:
  CaffeClassifier(const string& model_file,
             const string& trained_file,
             const bool use_GPU,
             const int batch_size);

  vector< vector<Prediction> > ClassifyBatch(const vector<Mat> imgs, int num_classes);
  vector<Blob<float>* > PredictBatch(const vector<Mat> imgs, float a, float b, float c);

  void SetMean(float a, float b, float c);
 private:

  void WrapBatchInputLayer(vector<vector<Mat> > *input_batch);

  void PreprocessBatch(const vector<Mat> imgs, vector<vector<Mat> >* input_batch);

 private:
  std::shared_ptr<Net<float> > net_;
  cv::Size input_geometry_;
  int num_channels_;
  int batch_size_;
  cv::Mat mean_;
  bool mean_128_;
};

CaffeClassifier::CaffeClassifier(const string& model_file,
                       const string& trained_file,
                       const bool use_GPU,
                       const int batch_size) {
   if (use_GPU) {
       Caffe::set_mode(Caffe::GPU);
       Caffe::SetDevice(0);
   }
   else
       Caffe::set_mode(Caffe::CPU);

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

  /* Load the binaryproto mean file. */
  //SetMean();
  mean_128_ = 0;
}

static bool PairCompare(const std::pair<float, int>& lhs,
                        const std::pair<float, int>& rhs) {
  return lhs.first > rhs.first;
}

/* Load the mean file in binaryproto format. */
void CaffeClassifier::SetMean(float a, float b, float c) {
  //mean_ = cv::Mat(input_geometry_, CV_32FC3, Scalar(128, 128, 128));
  //mean_ = cv::Mat(input_geometry_, CV_32FC3, Scalar(102.9801,115.9265,122.7717));
  mean_ = cv::Mat(input_geometry_, CV_32FC3, Scalar(a, b, c));
  if (abs(a-128) < 1 && abs(b-128)<1 && abs(c-128) < 1) {
    mean_128_ = 1;
  }
  //mean_ = cv::Mat(input_geometry_, CV_32FC3, Scalar(0,0,0));
}
//void CaffeClassifier::SetMean() {
//  //mean_ = cv::Mat(input_geometry_, CV_32FC3, Scalar(128, 128, 128));
//  mean_ = cv::Mat(input_geometry_, CV_32FC3, Scalar(102.9801,115.9265,122.7717));
//  //mean_ = cv::Mat(input_geometry_, CV_32FC3, Scalar(a, b, c));
//  //mean_ = cv::Mat(input_geometry_, CV_32FC3, Scalar(0,0,0));
//}

//std::vector< float >  CaffeClassifier::PredictBatch(const vector< cv::Mat > imgs) {
vector<Blob<float>* > CaffeClassifier::PredictBatch(const vector< cv::Mat > imgs, float a, float b, float c) {
  Blob<float>* input_layer = net_->input_blobs()[0];

  input_geometry_.height = imgs[0].rows;
  input_geometry_.width = imgs[0].cols;
  input_layer->Reshape(batch_size_, num_channels_,
                       input_geometry_.height,
                       input_geometry_.width);

  /* Forward dimension change to all layers. */
  SetMean(a, b, c);
  net_->Reshape();

  std::vector< std::vector<cv::Mat> > input_batch;
  WrapBatchInputLayer(&input_batch);

  PreprocessBatch(imgs, &input_batch);

  clock_t start = clock();
  net_->ForwardPrefilled();

  /* Copy the output layer to a std::vector */
  vector<Blob<float>* > outputs;
  for(int i = 0; i < net_->num_outputs(); i++) {
    Blob<float>* output_layer = net_->output_blobs()[i];
    outputs.push_back(output_layer);
  }
  return outputs;
}


void CaffeClassifier::WrapBatchInputLayer(std::vector<std::vector<cv::Mat> > *input_batch){
    Blob<float>* input_layer = net_->input_blobs()[0];

    int width = input_layer->width();
    int height = input_layer->height();
    int num = input_layer->num();
    float* input_data = input_layer->mutable_cpu_data();
    for ( int j = 0; j < num; j++){
        vector<cv::Mat> input_channels;
        for (int i = 0; i < input_layer->channels(); ++i){
          cv::Mat channel(height, width, CV_32FC1, input_data);
          input_channels.push_back(channel);
          input_data += width * height;
        }
        input_batch -> push_back(vector<cv::Mat>(input_channels));
    }
    //cv::imshow("bla", input_batch->at(1).at(0));
    //cv::waitKey(1);
}


void CaffeClassifier::PreprocessBatch(const vector<cv::Mat> imgs,
                                      std::vector< std::vector<cv::Mat> >* input_batch){

    //SetMean();
    for (int i = 0 ; i < imgs.size(); i++){
        cv::Mat img = imgs[i];
        std::vector<cv::Mat> *input_channels = &(input_batch->at(i));

        /* Convert the input image to the input image format of the network. */
        cv::Mat sample;
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

        cv::Mat sample_resized;
        if (sample.size() != input_geometry_)
          cv::resize(sample, sample_resized, input_geometry_);
        else
          sample_resized = sample;

        //imshow("debug.img", sample_resized);
        //waitKey(-1);

        cv::Mat sample_float;
        if (num_channels_ == 3)
          sample_resized.convertTo(sample_float, CV_32FC3);
        else
          sample_resized.convertTo(sample_float, CV_32FC1);


        cv::Mat sample_normalized;
        cv::subtract(sample_float, mean_, sample_normalized);

        //cv::Mat sample_rescaled;
        if (mean_128_) {
            cv::addWeighted(sample_normalized, 0.01, sample_normalized, 0, 0, sample_normalized);
        }

        /* This operation will write the separate BGR planes directly to the
         * input layer of the network because it is wrapped by the cv::Mat
         * objects in input_channels. */
        //cv::split(sample_normalized, *input_channels);
        //cv::split(sample_rescaled, *input_channels);

        cout << "input image " << sample_normalized.rows << " " << sample_normalized.cols << endl;
        
        cv::split(sample_normalized, *input_channels);

//        CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
//              == net_->input_blobs()[0]->cpu_data())
//          << "Input channels are not wrapping the input layer of the network.";
    }
}

struct Bbox {
    float confidence;
    Rect rect;
    Rect gt;
    bool deleted;

};
bool mycmp(struct Bbox b1, struct Bbox b2) {
    return b1.confidence > b2.confidence;
}


void nms(vector<struct Bbox>& p, float threshold) {
   sort(p.begin(), p.end(), mycmp);
   int cnt = 0;
   for(int i = 0; i < p.size(); i++) {
     if(p[i].deleted) continue;
     cnt += 1;
     for(int j = i+1; j < p.size(); j++) {
       if(!p[j].deleted) {
         cv::Rect intersect = p[i].rect & p[j].rect;
         float iou = intersect.area() * 1.0 / (p[i].rect.area() + p[j].rect.area() - intersect.area());
         if (iou > threshold) {
           p[j].deleted = true;
         }
       }
     }
   }
   //cout << "[after nms] left is " << cnt << endl;
}

int main(int argc, char** argv) {

  google::InitGoogleLogging(argv[0]);

  // caffe variables
  string prefix = ""; 
  string model_file   = prefix + string(argv[1]); //"../models/bvlc_googlenet/test.prototxt"; //  rcnn_googlenet.prototxt
  string trained_file = prefix + string(argv[2]); //"../output/default/voc_2015_train/googlenet_faster_rcnn_iter_48000.caffemodel"; // rcnn_googlenet.caffemodel

  string image_list = string(argv[3]); //"/home/zdb/data/test_roadside"; 

  CaffeClassifier classifier_single(model_file, trained_file, true, 1);

  model_file = string(argv[4]); //"/mnt/data1/zdb/data/car_detection/faster_rcnn/fcn_deepv/faster-rcnn-A80-D/caffe-fast-rcnn/caffe/models/car_not_car/deploy.prototxt"; // car_post_cls.prototxt
  trained_file = string(argv[5]); //"/mnt/data1/zdb/data/car_detection/faster_rcnn/fcn_deepv/faster-rcnn-A80-D/caffe-fast-rcnn/caffe/models/car_not_car/car_not_car_train_iter_30000.caffemodel"; // car_post_cls.caffemodel

  CaffeClassifier cls_single(model_file, trained_file, true, 1);

  FILE *fcin  = fopen(image_list.c_str(),"r");
  if(!fcin) {
    cout << "can not open filelist" << endl;
  }
  char image_filename[200];

  cout << "single mode" << endl;

  int tot_cnt = 0;

  string output_file = string(argv[6]);
  FILE* fid = fopen(output_file.c_str(),"w");
  while(fscanf(fcin, "%s", image_filename)!=EOF) {
        
        cout << "filename " << string(image_filename) << endl;
        vector<Mat> images;
        Mat image = imread(image_filename, -1);
        tot_cnt += 1;
        //if(tot_cnt > 100) break;
        Mat img;

        float border_ratio = 0.00;
        //copyMakeBorder(image,img,image.rows * border_ratio,image.rows * border_ratio,image.cols * border_ratio, image.cols*border_ratio, BORDER_CONSTANT, Scalar(103,116,123));

        img = image.clone();

        int max_size = max(img.rows, img.cols);
        int min_size = min(img.rows, img.cols);
        float target_min_size = 300.0;
        float target_max_size = 2000.0;
        float enlarge_ratio = target_min_size / min_size;

        if(max_size * enlarge_ratio > target_max_size) {
            enlarge_ratio = target_max_size / max_size;
        }

        int target_row = img.rows * enlarge_ratio;
        int target_col = img.cols * enlarge_ratio;

        cout << "[before]Image " << img.rows << " " << img.cols << endl;
        resize(img, img, Size(target_col, target_row));
        cout << "[after]Image " << img.rows << " " << img.cols << endl;



        //resize(img, img, Size(128,128));
        //img = img(Rect(5,5,118,118));
        images.push_back(img);

        int predict_batch_start = clock();
        vector<Blob<float>* > outputs = classifier_single.PredictBatch(images, 102.9801,115.9265,122.7717);
        cout << "PredictBatch Time: " << (clock() - predict_batch_start) * 1.0 / CLOCKS_PER_SEC << endl;

        Blob<float>* cls = outputs[0];
        Blob<float>* reg = outputs[1];
        
        const int scale_num = 21;
        cls->Reshape(cls->num()*scale_num, cls->channels()/scale_num, cls->height(), cls->width());
        reg->Reshape(reg->num()*scale_num, reg->channels()/scale_num, reg->height(), reg->width());

        assert(cls->num() == reg->num() && cls->num() == scale_num);

        assert(cls->channels() == 2);
        assert(reg->channels() == 4);

        assert(cls->height() == reg->height());
        assert(cls->width() == reg->width());

        
        vector<struct Bbox> vbbox;
        int cls_cnt = 0;
        int reg_cnt = 0;
        const float* cls_cpu = cls->cpu_data();
        const float* reg_cpu = reg->cpu_data();


        float mean[4] = {0, 0, 0, 0};
        //float mean[4] = {0.00010561, 0.01535682, 0.04042457, 0.06120514};
        //float std[4] = {0.13640414,0.1291149,0.25384777,0.24280438};
        //float std[4] = {0.13787029, 0.12806209, 0.24780841, 0.23739512};
        //float std[4] = {0.140, 0.133, 0.255, 0.246};
        float std[4] = {0.13848565, 0.13580033, 0.27823007, 0.26142551};

        float gt_ww[scale_num];
        float gt_hh[scale_num];
        float area[10] = {};
        for(int i = 0; i < scale_num / 3; i++) {
            area[i] = 50*50*pow(2, i);
        }
        float ratio[3] = {0.5, 1.0, 2.0}; // w / h
        int cnt = 0;


        //for(int i = 0; i < cls->count(); i++) {
        //    cout << i << " " << cls_cpu[i] << " " << cls_cpu[i+cls->width()*cls->height()]<< endl;
        //    if(i > 100) break;
        //}
        //return 0;

        float global_ratio = 1.0 * min(images[0].rows, images[0].cols) / target_min_size;
        for(int i = 0; i < scale_num/3; i++) {
            for(int j = 0; j < 3; j++) {
               gt_ww[cnt] = sqrt(area[i] * ratio[j]) * global_ratio;
               gt_hh[cnt] = gt_ww[cnt] / ratio[j] * global_ratio;
               cnt++; 
            }
        }
        cout << "gt_ww" << endl;
        for(int i = 0; i < scale_num; i++) {
            cout << gt_ww[i] << " ";
        }
        cout << endl << "gt_hh" << endl;
        for(int i = 0; i < scale_num; i++) {
            cout << gt_hh[i] << " ";
        }
        cout << endl;

        cout << "cls->height() " << cls->height() << endl;
        cout << "cls->width() " << cls->width() << endl;

        cout << "img->rows " << images[0].rows << endl;
        cout << "img->cols " << images[0].cols << endl;


        int post_process_start = clock();

        for(int h = 0; h < cls->height(); h++) {
            for(int w = 0; w < cls->width(); w++) {
                for(int i = 0; i < cls->num(); i++) {

                    float confidence = 0;
                    for(int j = 0; j < cls->channels(); j++) {
                        int cls_index = i; cls_index *= cls->channels();
                        cls_index += j; cls_index *= cls->height();
                        cls_index += h; cls_index *= cls->width();
                        cls_index += w;
                        //cout << "[debug] confidence j= " << j << " " <<  cls_cpu[cls_index] << " ";
                        if(j==1) {
                            float x1 = cls_cpu[cls_index];
                            float x0 = cls_cpu[cls_index - cls->height() * cls->width()];
                            //x1 -= min(x1, x0);
                            //x0 -= min(x1, x0);
                            confidence = exp(x1)/(exp(x1)+exp(x0));
                        }
                    }
                    //cout << endl;
                    //if(h==0 && w==0)
                    //cout << "[debug] confidence " << cls_cpu[cls_cnt] << " " << cls_cpu[cls_cnt+1] << endl;
                    float rect[4] = {};
                    //float gt_cx = (h+0.5) * 16.0; 
                    //float gt_cy = (w+0.5) * 16.0;
                    float gt_cx = w * 16.0; 
                    float gt_cy = h * 16.0;

                    

                    for(int j = 0; j < 4; j++) {
                        int reg_index = i; reg_index *= reg->channels();
                        reg_index += j; reg_index *= reg->height();
                        reg_index += h; reg_index *= reg->width();
                        reg_index += w;

                        rect[j] = reg_cpu[reg_index] *std[j] + mean[j];
                    }

                    rect[0] = rect[0] * gt_ww[i] + gt_cx;
                    rect[1] = rect[1] * gt_hh[i] + gt_cy;
                    rect[2] = exp(rect[2]) * gt_ww[i];
                    rect[3] = exp(rect[3]) * gt_hh[i];

                    if(confidence > 0.5) {
                        struct Bbox bbox;
                        bbox.confidence = confidence;
                        bbox.rect = Rect(rect[0]-rect[2]/2.0, rect[1]-rect[3]/2.0, rect[2], rect[3]);
                        bbox.rect &= Rect(0,0,images[0].cols, images[0].rows);
                        bbox.deleted = false;
                        bbox.gt = Rect(gt_cx - gt_ww[i]/2, gt_cy-gt_hh[i]/2, gt_ww[i], gt_hh[i]) & Rect(0,0,images[0].cols, images[0].rows);
                        vbbox.push_back(bbox);
                        
                        if(gt_cx - gt_ww[i]/2 >0 && gt_cy-gt_hh[i]/2 > 0 && gt_cx + gt_ww[i]/2 < images[0].cols && gt_cy + gt_hh[i]/2 < images[0].rows) {
                            //Mat img = images[0].clone();
                            //rectangle(img, Rect(gt_cx-gt_ww[i]/2, gt_cy-gt_hh[i]/2, gt_ww[i], gt_hh[i]), Scalar(0,0,255));
                            //rectangle(img, Rect(bbox.rect), Scalar(255,0,0));
                            //imshow("check", img);
                            //cout << "confidence " << bbox.confidence << endl;
                            //waitKey(100);

                        }
                    }
                } 
            }
        }
        /*
        for(int i = 0; i < cls->num(); i++) {
            for(int j = 0; j < cls->channels(); j++) {
            
            }
        }*/
        
        sort(vbbox.begin(), vbbox.end(), mycmp);

        nms(vbbox, 0.2);
        int tot_left = 0;
        //Mat resized_image = images[0].clone();
        
        Mat resized_image = images[0].clone();
        fprintf(fid, "%s ", image_filename);
        cout << "haha" << endl;
        for(int i = 0; i < vbbox.size(); i++) {
           if(!vbbox[i].deleted && vbbox[i].confidence > 0.8 && tot_left < 32) {
            
                
                
//              float cx = vbbox[i].gt.x + vbbox[i].gt.width / 2.0;
//              float cy = vbbox[i].gt.y + vbbox[i].gt.height / 2.0;
//
//              if (cx < border_ratio/(1.+2*border_ratio)*resized_image.cols || cx > (1.+border_ratio)/(1.+2*border_ratio)*resized_image.cols) continue;
//              if (cy < border_ratio/(1.+2*border_ratio)*resized_image.rows || cy > (1.+border_ratio)/(1.+2*border_ratio)*resized_image.rows) continue;
//
                Rect box = vbbox[i].rect;
                float x = box.x / enlarge_ratio, y=box.y / enlarge_ratio, w=box.width / enlarge_ratio, h=box.height / enlarge_ratio;
                x -= w*0.1;
                y -= h*0.1;
                w *= 1.2;
                h *= 1.2;
                Rect newbox(x, y, w, h);
                //Mat img = resized_image(newbox & Rect(0,0, resized_image.cols, resized_image.rows));
                Mat img = image(newbox & Rect(0,0, image.cols, image.rows));

                resize(img, img, Size(128,128));
                img = img(Rect(5,5,118,118));
                vector<Mat> images;
                images.push_back(img);
                vector<Blob<float>* > outputs = cls_single.PredictBatch(images, 128, 128, 128);

                Blob<float>* cls = outputs[0];
                const float* cls_cpu = cls->cpu_data();
                cout << "confs: id= " << i << " rpn " << vbbox[i].confidence << " car_no_car " << cls_cpu[0] << endl;
                //if (cls_cpu[0] > 0.5 && vbbox[i].confidence > 0.5) {
                if (cls_cpu[0] > 0.0 && vbbox[i].confidence > 0.5) {
                    rectangle(resized_image, vbbox[i].rect, Scalar(0,0,255)); // red
                    cout << "rpn confidence " << vbbox[i].confidence << endl;
                    tot_left ++;
                    //fprintf(fid, "%d %d %d %d %f ", vbbox[i].rect.x, vbbox[i].rect.y, vbbox[i].rect.width, vbbox[i].rect.height, vbbox[i].confidence);
                
                }
                if ((vbbox[i].confidence > 0.995 && cls_cpu[0] > 0.9) || vbbox[i].confidence > 0.999)
                    rectangle(resized_image, vbbox[i].rect, Scalar(255,0,0)); // red
           }
        }
        fprintf(fid, "\n");
        imshow("debug.jpg", resized_image);
        waitKey(-1);
        //char savename[1000];
        //sprintf(savename, "result/%d.png", tot_cnt);
        //imwrite(savename, resized_image);
        //waitKey(4);
        //cout << "tot_left " << tot_left << endl;

        cout << "post_process_time: " << (clock() - post_process_start) * 1.0 / CLOCKS_PER_SEC << endl;
  }
  return 0;
}
