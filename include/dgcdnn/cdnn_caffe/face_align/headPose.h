#pragma once
#include "posit.h"
using namespace std;
using namespace cv;
//head pose 
typedef struct 
{
	Mat rot;
	Mat t;
	float angles[3]; //pitch,yaw, roll
} HeadPose;

const float yaw_threshold = 15;
const float pitch_threshold = 10;
const int pointsNum = 39;
const float pts_threshold = 0.95;

static float featurePts_3DModel[pointsNum][3]=
{ // rotate to front face on 3d, then bilateral symmetry
	{-0.722191, 0.357883, -1.02562},
	{-0.681154, 0.140062, -1.01614},
	{-0.634217, -0.106129, -1.02095},
	{-0.56443, -0.354179, -0.908171},
	{-0.445352, -0.539988, -0.722865},
	{-0.282116, -0.666188, -0.541181},
	{0.00, -0.722471, -0.321038},//{0.00964536, -0.722471, -0.321038},
	{0.282116, -0.666188, -0.541181},
	{0.445352, -0.539988, -0.722865},
	{0.56443, -0.354179, -0.908171},
	{0.634217, -0.106129, -1.02095},
	{0.681154, 0.140062, -1.01614},
	{0.722191, 0.357883, -1.02562},
	{-0.502652, 0.487764, -0.322424},
	{-0.310034, 0.517126, -0.194245},
	{-0.104154, 0.497844, -0.138562},
	{-0.469554, 0.33427, -0.395509},
	{-0.318339, 0.341158, -0.304196},
	{-0.178226, 0.3224, -0.32474},
	{0.104154, 0.497844, -0.138562},
	{0.310034, 0.517126, -0.194245},
	{0.502652, 0.487764, -0.322424},
	{0.178226, 0.3224, -0.32474},
	{0.318339, 0.341158, -0.304196},
	{0.469554, 0.33427, -0.395509},
	{-0.104735, 0.345574, -0.284657},
	{-0.129597, 0.218243, -0.28145},
	{-0.170312, 0.100907, -0.265974},
	{-0.192748, -0.0758652, -0.262294},
	{0.192748, -0.0758652, -0.262294},
	{0.170312, 0.100907, -0.265974},
	{0.129597, 0.218243, -0.28145},
	{0.104735, 0.345574, -0.284657},
	{-0.289948, -0.34148, -0.336217},
	{-0.139081, -0.331302, -0.283398},
	{0.00, -0.338004, -0.205217},//{0.00460021, -0.338004, -0.205217},
	{0.139081, -0.331302, -0.283398},
	{0.289948, -0.34148, -0.336217},
	{0.0, -0.0189383, 0.0286134}//{-0.0190401, -0.0189383, 0.0286134}
};

void EstimateHeadPose(const Mat_<float>& shape,HeadPose& pose, vector<float> shape_score=vector<float>(), bool needmultiview = true);
void EstimateHeadPose_singleView(const Mat_<float>& shape,HeadPose& pose, vector<float> shape_score);
void EstimateHead_multiView(const Mat_<float>& shape,HeadPose& pose, vector<float> shape_score);
void EstimateHeadPose_Posit( const Mat_<float>& shape, HeadPose& pose, vector<int> pts_id);
void get2DshapeCorres3D(const Mat_<float>& shape, Mat_<float>& reShape);
void get2DScoreCorres3D(const vector<float>& shape_score, vector<float>& reShape_score);
void getValidPts(const vector<float>& shape_score, const float& pts_threshold, int pts_Num, int pts_id[], vector<int>& shape_id);
