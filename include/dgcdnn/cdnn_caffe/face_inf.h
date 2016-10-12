#ifndef FACE_INF_H 
#define FACE_INF_H
#define BITS_PIXEL 8
#define MIN_FACE_SIZE 0
#define BEST_FACE_SIZE 60

#include "CascadeLight.h"
#include "face_para.h"
#include <opencv/cv.h>

//using namespace cv;
#define MIN_CROP_SIZE 150
#define MULTIPATCH 1 //set it if use multipatch

#define REF_IMAGE_WIDTH 150
#define REF_LEFT_EYE_X    51
#define REF_EYE_DISTRANCE 48
#define REF_EYE_Y    64
#define REF_MOUTH_CENTER 114

struct Point_2d_f {
  float x;
  float y;
};
struct RefImageInfo
{
  int m_ref_lefteye_x;
  int m_ref_lefteye_y;
  int m_ref_righteye_x;
  int m_ref_righteye_y;
  int m_ref_centermouth_x;
  int m_ref_centermouth_y;
  int m_ref_height;
  int m_ref_width;
  int m_ref_channel;
};
struct DetectedFaceInfo
{
  float left;
  float top;
  float width;
  float height;
  float conf;
  int degree;
  int pose;
};
struct LandMarkInfo
{
  Point_2d_f lefteye;
  Point_2d_f righteye;
  Point_2d_f nose;
  Point_2d_f mouth;
  vector<Point_2d_f> landmarks;
  vector<float> landmark_scores;
  double score;
  float yaw; // in degree
  float pitch; // in degree
  float roll; // in degree
};


struct vis_FaceInfo
{
  string PicId;
  string FaceId;
  string product;
  string name;
  bool IsDetected;
  bool IsAlign;
  bool IsParsed;
  bool HasFeature;
  bool HasBlur;
  IplImage* refimage;
  DetectedFaceInfo detectFaceInfo;    
  LandMarkInfo  landMarkInfo;
  vector<char> feature;
  char* cropImage;
  int cropImageLen;
  vis_FaceInfo() {
    IsDetected = false;
    IsAlign = false;
    IsParsed = false;
    HasFeature = false;
    HasBlur = false;
    cropImage = NULL;
    cropImageLen = 0;
    refimage = NULL;
    feature.clear();
    memset(&detectFaceInfo, 0, sizeof(DetectedFaceInfo));
    memset(&landMarkInfo, 0, sizeof(LandMarkInfo));
  }
};
struct FaceLocInfo{
  int   width;
  int   height;
  int   left;
  int top;
  int face_width;
  int face_height;
};  
struct attr_info
{
  vector<float> feature;
  int tag;
  string id;
};
void releaseFaceInfo(vis_FaceInfo& faceinfo);
struct BDFACEImageInfo
{
  char* imageInfo;
  int imageInfoLen;
  string product;
  string PicId;
  IplImage* image;
  BDFACEImageInfo()
  {
    imageInfo = NULL;
    imageInfoLen = 0;
    //PicId = NULL;
    image = NULL;
  }
};
enum face_op
{
  NONE = 0,
  DETECT = 1,
  ALIGN = 2,
  EXTRACT_FEATURE = 4,
  CALC_SMILE = 8,
  CALC_SEX = 16,
  CALC_BEAUTY = 32,
  CALC_AGE = 64,
  REMOVE_TEXTURE = 128,
  GET_REF_IMAGE = 256,
  CALC_FEATURE_SIM = 512,
  CALC_RACE = 1024,
  CALC_GLASS = 2048,
  PARSING = 4096,
  CALC_BLUR = 8192 
};
enum FACETYPE
{
  FACETYPE_NORMAL = 0,
  FACETYPE_IDPHOTO = 1,
  FACETYPE_LIGHTTEXTURE = 2,
  FACETYPE_HARDTEXTURE = 3,
  FACETYPE_NUM = 4 //number of types
};
enum err_no
{
  //init error
  FACEAPI_SUCCEED = 0,
  INIT_CONFIG_ERROR = 101,
  INIT_DETECTOR_ERROR = 102,
  INIT_ALIGN_ERROR = 103,
  INIT_PARSING_ERROR = 104,
  INIT_FEATURE_ERROR = 105,
  INIT_WARP_ERROR = 106,
  INIT_SMILE_ERROR = 107,
  INIT_GENDER_ERROR = 108,
  INIT_BEAUTY_MALE_ERROR = 109,
  INIT_BEAUTY_FEMALE_ERROR = 110,
  INIT_AGE_ERROR  = 111,
  INIT_GLASS_ERROR = 112,
  INIT_STAR_ERROR = 113,
  INIT_RACE_ERROR = 114,
  INIT_ATTR_ERROR = 115,
  INIT_FEATURE_METRICMODEL_ERROR = 116,
  INIT_FEATURE_CDNNMODEL_ERROR = 117,
  INIT_FEATURE_CONFIG_CDNNNUM_ERROR = 118,
  INIT_FEATURE_CONFIG_FILENOTEXIST_ERROR = 119,
  INIT_COMPARE_NULL_ERROR = 120,
  INIT_COMPARE_FILE_NOTEXIST_ERROR = 121,
  INIT_REMOVE_TEXTURE_ERROR = 122,
  INIT_BLUR_ERROR = 123,
  //service parameter error
  SERVICE_INPUT_IMAGE_EMPTY_ERROR = 201,
  SERVICE_INPUT_IMAGE_BASE64_ERROR = 202,
  SERVICE_INPUT_CALCTYPE_EMPTY_ERROR = 203,
  SERVICE_INPUT_CALCTYPE_PARSE_ERROR = 204,
  SERVICE_INPUT_FACEINFO_ARRAY_ERROR = 205,
  SERVICE_INPUT_IMAGE_READ_ERROR = 206,
  SERVICE_INPUT_FACEDATA1_EMPTY_ERROR = 207,
  SERVICE_INPUT_FACEDATA2_EMPTY_ERROR = 208,
  SERVICE_INPUT_FEATURE1_BASE64_ERROR = 209,
  SERVICE_INPUT_FACERET1_EMPTY_ERROR = 210,
  SERVICE_INPUT_FEATURE1_EMPTY_ERROR = 211,
  SERVICE_INPUT_FEATURE2_EMPTY_ERROR = 212,
  SERVICE_INPUT_FACERET2_EMPTY_ERROR = 213,
  SERVICE_INPUT_FEATURE2_BASE64_ERROR = 214,
  SERVICE_INPUT_FACENUM_NOT_EXIST_ERROE = 215,
  SERVICE_INPUT_FACE_NOT_EXIST_ERROE = 216,
  SERVICE_INPUT_FEATURE1_PROTO_ERROR = 217,

  //SDK parameter error
  PROCESS_INPUT_IMAGE_EMPTY_ERROR = 301,
  PROCESS_DETECTOR_EMPTY_ERROR = 302,
  PROCESS_ALIGN_EMPTY_ERROR = 303,
  PROCESS_PARSING_EMPTY_ERROR = 304,
  PROCESS_FEATURE_EMPTY_ERROR = 305,
  PROCESS_STAR_EMPTY_ERROR = 306,
  PROCESS_WARP_EMPTY_ERROR = 307,
  PROCESS_GENDER_EMPTY_ERROR = 308,
  PROCESS_RACE_EMPTY_ERROR = 309,
  PROCESS_GLASS_EMPTY_ERROR = 310,
  PROCESS_EXPRESSION_EMPTY_ERROR = 311,
  PROCESS_AGE_EMPTY_ERROR = 312,
  PROCESS_BEAUTY_EMPTY_ERROR = 313,
  PROCESS_TEXTURE_CLASSIFY_EMPTY_ERROR = 314,
  PROCESS_TEXTURE_REMOVER_EMPTY_ERROR = 315,
  PROCESS_INPUT_IMG_TOO_SMALL_ERROR = 316,
  PROCESS_BLUR_EMPTY_ERROR = 317,
  //calcsim error
  CALCSIM_WRONG_INPUT_ERROR = 401,
  CALCSIM_MISMATCH_INPUT_ERROR = 402,
  CALCSIM_UNSURPORTED_TYPE_ERROR = 403,
  CALCSIM_NOT_INITIALIZED_ERROR = 404,
  //idlapi interface
  IDLAPI_CMDID_NOTSUPPORT = 501

};
  template<class T>
int calc_face_dist(T* feature1,T* feature2,int n,T& sim)
{               
  if ((!feature1) || (!feature2) || n < 1)
  {
    return -1;
  }
  sim = 0.0;
  for (int i = 0; i < n; i++)
  {
    sim += (feature1[i] - feature2[i]) * (feature1[i] - feature2[i]);
  }
  return 0;
}       
void sort_face_by_size(vector<DetectedFaceInfo>& faceinfo);

bool init_knn_model(const char* filename,vector<attr_info>& model);

bool NameListInit(const char* filename,vector<string>& namelist);

bool LocInfoInit(const char* filename,map<string,FaceLocInfo>& locinfo);
/*
BDFACEImageInfo    :        input image
face_op        :        operation type
face_cascade    :        detect handler
face_aligner    :        align handler
face_metric    :        metric handler
faceInfo    :        output face results
src_size    :        output image longest size    
bias        :        bias for refimage size
needrotate    :        whether need 4 direction or not
needcrop    :        whetther need crop face image or not
max_disp_face_size:        max displaying face size
quality        :        jpeg quality of displaying face
need_compress    :        whether need compress input image or not
face_ratio    :        least face ratio of input image to detect
min_face_size    :        min face size to detect
gif_frame    :        frames to process for gif image
isRaw        :        whether extract 18000 raw feature or not
*/
/*int BDFACEProcess(BDFACEImageInfo src_info,face_op op_type,SCascadeL& face_cascade, CFaceAligner& face_aligner,CCovMatMetric& face_metric,vector<vis_FaceInfo>& faceInfo, int& src_size,int bias = 0,bool needrotate = false,bool needcrop=false,int max_disp_face_size = 350,int quality = 75,bool need_compress=false,double face_ratio = 0.0,int min_face_size = MIN_FACE_SIZE, int gif_frame = 1,bool isRaw = false);

  int GetFeatureFromLandmark(BDFACEImageInfo src_info,CCovMatMetric& face_metric,LandMarkInfo landmark,vector<float>& feature);
  */
CvRect GetCropRect(int width,int height,CvRect rect,bool isCenter, double ratio);

IplImage* cvGetSubImage(IplImage *image, CvRect roi, int dst_width,int dst_height);

bool deserialize(char* filename,vector<vis_FaceInfo>& faceInfo);
bool serialize(char* filename,vector<vis_FaceInfo>& faceInfo);
int cvRotateImage(IplImage* src,IplImage** dst,int degree); 
float BeautyNormalize(float score, float max,float min);
int cdnn_predict(IplImage* img, int dstsize,int cropborder, vector<float>& pts, vector<int>& scale, int patchsize,float* data_mean,void *model,float* probs);
int cdnn_extract_feature(IplImage* img, int dstsize,int cropborder, vector<float>& pts, vector<int>& scale, int patchsize,float* data_mean,void *model,vector<string>& outlayer,float*& probs,int& featlen);
int LoadParam(const char* pConfDir,const char* pConfName,FacePara& param);
int get_face_input_json(string input, string &imageid, IplImage**& image, int& img_num, 
    int& logid, int& opType, vector<vis_FaceInfo>& faceInfo, int& targetFace, 
    int&imglen1, int& is_multi_metric);
int get_cmdid_json(string input);
void get_face_output_json(int errno, IplImage* image, string imageid, int opType,
    vector<vis_FaceInfo>& faceInfo, string& output);
void get_face_output_proto(int errno, IplImage* image, string imageid, int opType, 
    vector<vis_FaceInfo>& faceInfo, string& output);
int get_compare_input_json(string input, char*& feature1, int& featlen1, 
    char*& feature2, int& featlen2);
int get_compare_input_proto(string input, string& imageid1, 
    string& imageid2, char*& feature1, int& featlen1, char*& feature2, int& featlen2);
int get_compare_output_json(int ret, string imageid1, string imageid2,
    float score, string& output);
int encode_feature(float* feature, int featlen);
int decode_feature(float* feature, int featlen);

#endif

