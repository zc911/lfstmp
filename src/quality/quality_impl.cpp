#include <quality/qual_blurm.h>
#include <quality/qual_frontalm.h>
#include <quality/qual_posem.h>
#include <config.h> 
#include <stdexcept>
#include <string>
#include <memory>
#include <stdint.h>

using namespace cv;
using namespace std;
namespace DGFace{

/*======================= blur_metric quality ========================= */
BlurMQuality::BlurMQuality(void) {
}

BlurMQuality::~BlurMQuality(void) {
}

float BlurMQuality::blur_metric(const Mat &image, short *sobelTable)
{
    int i, j, mul;
    int width  = image.cols;
    int height = image.rows;
    const uint8_t *data = image.ptr();

    for(i = 1, mul = i * width; i < height - 1; i++, mul += width)
          for(j = 1; j < width - 1; j++)
            sobelTable[mul+j] = abs(data[mul+j-width-1] + 2*data[mul+j-1] + data[mul+j-1+width] -
                        data[mul+j+1-width] - 2*data[mul+j+1] - data[mul+j+width+1]);

    for(i = 1, mul = i*width; i < height - 1; i++, mul += width)
          for(j = 1; j < width - 1; j++)
            if(sobelTable[mul+j] < 50/* || sobelTable[mul+j] <= sobelTable[mul+j-1] ||\
                        sobelTable[mul+j] <= sobelTable[mul+j+1]*/) sobelTable[mul+j] = 0;
    int totLen = 0;
    int totCount = 1;

    unsigned char suddenThre = 50;
    unsigned char sameThre = 3;

    for(i = 1, mul = i*width; i < height - 1; i++, mul += width)
    {
        for(j = 1; j < width - 1; j++)
        {
            if(sobelTable[mul+j])
            {
                int   count = 0;
                int      t;
                unsigned char tmpThre = 5;
                unsigned char max = data[mul+j] > data[mul+j-1] ? 0 : 1;

                for(t = j; t > 0; t--)
                {
                    count++;
                    if(abs(data[mul+t] - data[mul+t-1]) > suddenThre)
                        break;

                    if(max && data[mul+t] > data[mul+t-1])
                        break;
                   

                    if(!max && data[mul+t] < data[mul+t-1])
                        break;

                    int tmp = 0;
                    for(int s = t; s > 0; s--)
                    {
                        if(abs(data[mul+t] - data[mul+s]) < sameThre)
                        {
                            tmp++;
                            if (tmp <= tmpThre)
                                continue;
                        }
                        break;
                    }

                    if(tmp > tmpThre) break;
                }

                max = data[mul+j] > data[mul+j+1] ? 0 : 1;

                for(t = j; t < width; t++)
                {
                    count++;
                    if(abs(data[mul+t] - data[mul+t+1]) > suddenThre)
                        break;

                    if((max && data[mul+t] > data[mul+t+1]) || (!max && data[mul+t] < data[mul+t+1]))
                        break;

                    int tmp = 0;
                    for(int s = t; s < width; s++)
                    {
                        if(abs(data[mul+t] - data[mul+s]) < sameThre)
                        {
                            tmp++;
                            if (tmp <= tmpThre)
                                continue;
                        }
                        break;
                    }
                    if(tmp > tmpThre) break;
                }
                count--;
                totCount++;
                totLen += count;
            }
        }
    }
    float result = static_cast<float>(totLen)/totCount;
    return result;
}

float BlurMQuality::quality(const Mat &image) {
    Mat sample;
    if (image.channels() == 3)
        cvtColor(image, sample, CV_BGR2GRAY);
    else if (image.channels() == 4)
        cvtColor(image, sample, CV_BGRA2GRAY);
    unique_ptr<short> sobelTable(new short[sample.cols * sample.rows]);
    return blur_metric(sample, sobelTable.get());
}


/*======================= Frontal face quality ========================= */
FrontalMQuality::FrontalMQuality(void) : _detector(new DlibDetector(60, 45)) {
}

FrontalMQuality::~FrontalMQuality(void) {
    delete _detector;
}

float FrontalMQuality::quality(const Mat &image) {
    float frontal_score = 0.f;
    vector<DetectResult> fine_det_results;
    vector<Mat> images;
    images.push_back(image);
    
    _detector->detect(images, fine_det_results);
    if (fine_det_results[0].boundingBox.size()) {
        frontal_score = fine_det_results[0].boundingBox[0].first; // dlib det score
    }
    return frontal_score;
}

/*====================== pose quality ======================== */
PoseQuality::PoseQuality(void) {}

PoseQuality::~PoseQuality(void) {}

vector<float> PoseQuality::quality(const AlignResult &align_result) {
		//calculating the head pose
		Mat_<float> s(align_result.landmarks.size()*2,1);

		for(size_t i=0; i<align_result.landmarks.size(); ++i)
		{       
			s(i,0) = align_result.landmarks[i].x;
			s(i+align_result.landmarks.size(),0) = align_result.landmarks[i].y;
		}       
		HeadPose pose;
		EstimateHeadPose(s,pose);   //head pose estimation 

		vector<float> pose_angles;
		pose_angles.resize(sizeof(pose.angles)/sizeof(pose.angles[0]));
        pose_angles[0] = pose.angles[0];//pitch
        pose_angles[1] = pose.angles[1];//yaw
        pose_angles[2] = pose.angles[2];//roll
		return pose_angles;
}

/*====================== select detector ======================== */
Quality *create_quality(const string &prefix) {
    Config *config    = Config::instance();
    string type       = config->GetConfig<string>(prefix + "quality", "blurm");

    if (type == "blurm")
        return new BlurMQuality();
    else if (type == "frontalm")
        // create dlib frontal face detector
        return new FrontalMQuality();
	else if (type == "posem")
	    // create pose estimation
		return new PoseQuality();
    throw new runtime_error("unknown quality measure");
}
Quality *create_quality(const std::string& method, const std::string& model_dir, int gpu_id) {
	if (method == "blurm")
        return new BlurMQuality();
    else if (method == "frontalm")
        // create dlib frontal face detector
        return new FrontalMQuality();
	else if (method == "posem")
	    // create pose estimation
		return new PoseQuality();
    throw new runtime_error("unknown quality measure");
}
}
