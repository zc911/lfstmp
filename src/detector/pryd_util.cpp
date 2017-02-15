#include <opencv2/opencv.hpp>
#include <fstream>
#include "detector/pryd_util.hpp"
namespace db {

using namespace std;
#ifndef ELLISION
#define ELLISION  1e-9
#endif

void GetBBoxes(const vector<float>& coords, const int key_points_count,
		vector<vector<float> >& bboxes) {

	//CHECK(coords.size() % (key_points_count * 2) == 0)
	//		<< "The number of key points is wrong.";

	bboxes = vector<vector<float> >(coords.size() / (key_points_count * 2), vector<float>(4, -1));
	for (int i = 0; i < bboxes.size(); ++i) {
		float min_x = -1, max_x = -1;
		float min_y = -1, max_y = -1;
		for (int j = 0; j < key_points_count; ++j) {
			int idx = i * key_points_count + j;
			idx *= 2;
			if (std::abs(coords[idx] - (-1)) < ELLISION
					|| std::abs(coords[idx + 1] - (-1)) < ELLISION) continue;

			if (std::abs(min_x - (-1)) < ELLISION) {
				min_x = coords[idx];
				max_x = coords[idx];

				min_y = coords[idx + 1];
				max_y = coords[idx + 1];
			} else {
				min_x = MIN(min_x, coords[idx]);
				max_x = MAX(max_x, coords[idx]);

				min_y = MIN(min_y, coords[idx + 1]);
				max_y = MAX(max_y, coords[idx + 1]);
			}
		}
		bboxes[i][0] = min_x;
		bboxes[i][1] = min_y;

		bboxes[i][2] = max_x;
		bboxes[i][3] = max_y;
	}
}

void GetBBoxStandardScale(const vector<float>& coords, const int key_points_count,
		const int standard_bbox_diagonal_len, vector<float>& standard_scale) {

	standard_scale.clear();

	vector<vector<float> > bboxes;
	GetBBoxes(coords, key_points_count, bboxes);
	for (int j = 0; j < bboxes.size(); ++j) {
		if (std::abs(bboxes[j][0] - (-1)) < ELLISION) {
			standard_scale.push_back(1);
		} else {
			float w = bboxes[j][0] - bboxes[j][2];
			float h = bboxes[j][1] - bboxes[j][3];
			standard_scale.push_back(standard_bbox_diagonal_len / std::sqrt(w * w + h * h));
		}
	}
}

void GetAllBBoxStandardScale(const vector<std::pair<std::string, vector<float> > >& samples,
		const int key_points_count, const int standard_bbox_diagonal_len,
		vector<vector<float> >& bboxes_standard_scale) {

	bboxes_standard_scale = vector<vector<float> >(samples.size(), vector<float>());
	for (int i = 0; i < samples.size(); ++i) {
		GetBBoxStandardScale(samples[i].second, key_points_count, standard_bbox_diagonal_len,
				bboxes_standard_scale[i]);
	}
}


template <typename Dtype>
bool compareCandidate(const pair<Dtype, vector<float> >& c1,
    const pair<Dtype, vector<float> >& c2) {
  return c1.first >= c2.first;
}

template bool compareCandidate<float>(const pair<float, vector<float> >& c1,
    const pair<float, vector<float> >& c2);
template bool compareCandidate<double>(const pair<double, vector<float> >& c1,
    const pair<double, vector<float> >& c2);

template <typename Dtype>
bool compareCandidate_v2(const vector<Dtype>  & c1,
    const  vector<Dtype>  & c2) {
  return c1[0] >= c2[0];
}

template bool compareCandidate_v2(const vector<float>  & c1,
    const  vector<float>  & c2);
template bool compareCandidate_v2(const vector<double>  & c1,
    const  vector<double>  & c2);

template <typename Dtype>
const vector<bool> nms(vector<pair<Dtype, vector<float> > >& candidates,
    const float overlap, const int top_N, const bool addScore) {
  vector<bool> mask(candidates.size(), false);

  if (mask.size() == 0) return mask;

  vector<bool> skip(candidates.size(), false);
  std::stable_sort(candidates.begin(), candidates.end(), compareCandidate<Dtype>);

  vector<float> areas(candidates.size(), 0);
  for (int i = 0; i < candidates.size(); ++i) {
  	areas[i] = (candidates[i].second[2] - candidates[i].second[0] + 1)
				* (candidates[i].second[3] - candidates[i].second[1] + 1);
  }

  for (int count = 0, i = 0; count < top_N && i < mask.size(); ++i) {
    if (skip[i]) continue;

    mask[i] = true;
    ++count;

    // suppress the significantly covered bbox
    for (int j = i + 1; j < mask.size(); ++j) {
      if (skip[j]) continue;

      // get intersections
      float xx1 = MAX(candidates[i].second[0], candidates[j].second[0]);
      float yy1 = MAX(candidates[i].second[1], candidates[j].second[1]);
      float xx2 = MIN(candidates[i].second[2], candidates[j].second[2]);
      float yy2 = MIN(candidates[i].second[3], candidates[j].second[3]);
      float w = xx2 - xx1 + 1;
      float h = yy2 - yy1 + 1;
      if (w > 0 && h > 0) {
        // compute overlap
    	float o = w * h / std::min(areas[j],areas[i]);
        //float o = w * h / areas[j];
        if (o > overlap) {
          skip[j] = true;

          if (addScore) {
          	candidates[i].first += candidates[j].first;
          }
        }
      }
    }
  }

  return mask;
}

template const vector<bool> nms<float> (vector<pair<float, vector<float> > >& candidates,
    const float overlap, const int top_N, const bool addScore);
template const vector<bool> nms<double> (vector<pair<double, vector<float> > >& candidates,
		const float overlap, const int top_N, const bool addScore);

template <typename Dtype>
const vector<bool> bbox_voting(vector< vector<Dtype> >& candidates,
    const Dtype overlap){

	  vector<bool> mask(candidates.size(), false);
	  vector<int> voted_weight(candidates.size(), 1);

	  if (mask.size() == 0) return mask;
	  //LOG(INFO)<<"overlap: "<<overlap;
	  vector<bool> skip(candidates.size(), false);
	  std::stable_sort(candidates.begin(), candidates.end(), compareCandidate_v2<Dtype>);

	  vector<Dtype> areas(candidates.size(), 0);
	  for (int i = 0; i < candidates.size(); ++i) {
	  	areas[i] = (candidates[i][3] - candidates[i][1] + 1)
					* (candidates[i][4] - candidates[i][2] + 1);
	  }

	  for (int count = 0, i = 0;   i < mask.size(); ++i) {
	    if (skip[i]) continue;

	    mask[i] = true;
	    ++count;

	    // suppress the significantly covered bbox
	    for (int j = i + 1; j < mask.size(); ++j) {
	      if (skip[j]) continue;

	      // get intersections
	      Dtype xx1 = MAX(candidates[i][1], candidates[j][1]);
	      Dtype yy1 = MAX(candidates[i][2], candidates[j][2]);
	      Dtype xx2 = MIN(candidates[i][3], candidates[j][3]);
	      Dtype yy2 = MIN(candidates[i][4], candidates[j][4]);
	      Dtype w = xx2 - xx1 + 1;
	      Dtype h = yy2 - yy1 + 1;
	      //LOG(INFO)<<"xx1:"<<xx1<<"  yy1:"<<yy1<<"  xx2:"<<xx2<<"  yy2:"<<yy2;
	      if (w > 0 && h > 0) {
	        // compute overlap
	    	Dtype o = w * h / (areas[j]+areas[i] - w * h);
	       // LOG(INFO)<<o;
	        if (o > overlap) {
	        	skip[j] = true;
	          	candidates[i][0] += candidates[j][0];
	          	voted_weight[i]+= 1;
	          	candidates[i][1] = (candidates[j][1] + candidates[i][1] * (voted_weight[i] - 1))/voted_weight[i];
	          	candidates[i][2] = (candidates[j][2] + candidates[i][2] * (voted_weight[i] - 1))/voted_weight[i];
	          	candidates[i][3] = (candidates[j][3] + candidates[i][3] * (voted_weight[i] - 1))/voted_weight[i];
	          	candidates[i][4] = (candidates[j][4] + candidates[i][4] * (voted_weight[i] - 1))/voted_weight[i];
	        }
	      }

	    }

	  }

	  return mask;
}

template const vector<bool> bbox_voting(vector< vector<float> >& candidates,
    const float overlap);
template const vector<bool> bbox_voting(vector< vector<double> >& candidates,
    const double overlap);


template <typename Dtype>
const vector<bool> nms(vector < vector<Dtype>   >& candidates,
    const Dtype overlap, const int top_N, const bool addScore) {

  vector<bool> mask(candidates.size(), false);

  if (mask.size() == 0) return mask;
  //LOG(INFO)<<"overlap: "<<overlap;
  vector<bool> skip(candidates.size(), false);
  std::stable_sort(candidates.begin(), candidates.end(), compareCandidate_v2<Dtype>);

  vector<Dtype> areas(candidates.size(), 0);
  for (int i = 0; i < candidates.size(); ++i) {
  	areas[i] = (candidates[i][3] - candidates[i][1] + 1)
				* (candidates[i][4] - candidates[i][2] + 1);
  }

  for (int count = 0, i = 0; count < top_N && i < mask.size(); ++i) {
    if (skip[i]) continue;

    mask[i] = true;
    ++count;

    // suppress the significantly covered bbox
    for (int j = i + 1; j < mask.size(); ++j) {
      if (skip[j]) continue;

      // get intersections
      Dtype xx1 = MAX(candidates[i][1], candidates[j][1]);
      Dtype yy1 = MAX(candidates[i][2], candidates[j][2]);
      Dtype xx2 = MIN(candidates[i][3], candidates[j][3]);
      Dtype yy2 = MIN(candidates[i][4], candidates[j][4]);
      Dtype w = xx2 - xx1 + 1;
      Dtype h = yy2 - yy1 + 1;
      //LOG(INFO)<<"xx1:"<<xx1<<"  yy1:"<<yy1<<"  xx2:"<<xx2<<"  yy2:"<<yy2;
      if (w > 0 && h > 0) {
        // compute overlap
    	Dtype o = w * h / std::min(areas[j],areas[i]);
       // LOG(INFO)<<o;
        if (o > overlap) {
          skip[j] = true;

          if (addScore) {
          	candidates[i][0] += candidates[j][0];
          }
        }
      }
    }
  }
  return mask;
}

template const vector<bool> nms (vector < vector<float>   >& candidates,
    const float overlap, const int top_N, const bool addScore);

template const vector<bool> nms (vector < vector<double>   >& candidates,
    const double overlap, const int top_N, const bool addScore);


template <typename Dtype>
const vector<bool> nms(vector< BBox<Dtype> >& candidates,
    const Dtype overlap, const int top_N, const bool addScore) {
  vector<bool> mask(candidates.size(), false);

  if (mask.size() == 0) return mask;

  vector<bool> skip(candidates.size(), false);
  std::stable_sort(candidates.begin(), candidates.end(), BBox<Dtype>::greater);

  vector<float> areas(candidates.size(), 0);
  for (int i = 0; i < candidates.size(); ++i) {
  	areas[i] = (candidates[i].x2 - candidates[i].x1 + 1)
				* (candidates[i].y2- candidates[i].y1 + 1);
  }

  for (int count = 0, i = 0; count < top_N && i < mask.size(); ++i) {
    if (skip[i]) continue;

    mask[i] = true;
    ++count;

    // suppress the significantly covered bbox
    for (int j = i + 1; j < mask.size(); ++j) {
      if (skip[j]) continue;

      // get intersections
      float xx1 = MAX(candidates[i].x1, candidates[j].x1);
      float yy1 = MAX(candidates[i].y1, candidates[j].y1);
      float xx2 = MIN(candidates[i].x2, candidates[j].x2);
      float yy2 = MIN(candidates[i].y2, candidates[j].y2);
      float w = xx2 - xx1 + 1;
      float h = yy2 - yy1 + 1;
      if (w > 0 && h > 0) {
        // compute overlap
        float o = w * h / areas[j];
        if (o > overlap) {
          skip[j] = true;

          if (addScore) {
          	candidates[i].score += candidates[j].score;
          }
        }
      }
    }
  }

  return mask;
}
template const vector<bool> nms  (vector< BBox<float> >&  candidates,
    const float overlap, const int top_N, const bool addScore );
template const vector<bool> nms  (vector< BBox<double> >&  candidates,
		const double overlap, const int top_N, const bool addScore );

/*
 *
 *
 * Designed by Dukang for Roteated Rect
 *
 */
template <typename Dtype>
Dtype getApproximateArea(const  RotateBBox<Dtype>& rotateSquareRect)
{
    return 3.14159265* (  (rotateSquareRect.lt_x- rotateSquareRect.rb_x) * (rotateSquareRect.lt_x- rotateSquareRect.rb_x)
                                              +(rotateSquareRect.lt_y- rotateSquareRect.rb_y) * (rotateSquareRect.lt_y- rotateSquareRect.rb_y) ) *3/16;
    /*
    return ( sqrt( (rotateSquareRect.rt_x-rotateSquareRect.lt_x)*(rotateSquareRect.rt_x-rotateSquareRect.lt_x)
                +(rotateSquareRect.rt_y-rotateSquareRect.lt_y)*(rotateSquareRect.rt_y-rotateSquareRect.lt_y) )
            *sqrt( (rotateSquareRect.lb_x-rotateSquareRect.lt_x)*(rotateSquareRect.lb_x-rotateSquareRect.lt_x) 
                +(rotateSquareRect.lb_y-rotateSquareRect.lt_y)*(rotateSquareRect.lb_y-rotateSquareRect.lt_y) ) );

                */
}

template <typename Dtype>
Dtype getInterSectinArea_Circle( const Dtype center_1_x, const Dtype center_1_y, const Dtype radius_1,
                                 const Dtype center_2_x, const Dtype center_2_y, const Dtype radius_2 )
{

    Dtype center_dist = sqrt( (center_1_x -center_2_x) * (center_1_x -center_2_x) + (center_1_y -center_2_y) * (center_1_y -center_2_y) );

    Dtype cos_half_angle_1 = (center_dist*center_dist + radius_1*radius_1 - radius_2*radius_2)/(2*center_dist*radius_1);
    Dtype cos_half_angle_2 = (center_dist*center_dist + radius_2*radius_2 - radius_1*radius_1)/(2*center_dist*radius_2);

    Dtype angle_1 = acos(cos_half_angle_1)*2;
    Dtype angle_2 = acos(cos_half_angle_2)*2;

    Dtype circum_circle_Area = radius_1*radius_1*(angle_1-sin(angle_1))/2 + radius_2*radius_2*(angle_2-sin(angle_2))/2;
    circum_circle_Area = std::max( (Dtype)0, circum_circle_Area );
    return circum_circle_Area;

}

template < typename Dtype> 
Dtype getApproximateInterSectionArea( const RotateBBox<Dtype>& rotateSquareRect_1, const RotateBBox<Dtype>& rotateSquareRect_2)
{
    Dtype center_1_x = rotateSquareRect_1.lt_x/2 + rotateSquareRect_1.rb_x/2;
    Dtype center_1_y = rotateSquareRect_1.lt_y/2 + rotateSquareRect_1.rb_y/2;
    Dtype circumcircle_radius_1 = sqrt(  (rotateSquareRect_1.lt_x- rotateSquareRect_1.rb_x) * (rotateSquareRect_1.lt_x- rotateSquareRect_1.rb_x)
                                        +(rotateSquareRect_1.lt_y- rotateSquareRect_1.rb_y) * (rotateSquareRect_1.lt_y- rotateSquareRect_1.rb_y) )/2;

    Dtype center_2_x = rotateSquareRect_2.lt_x/2 + rotateSquareRect_2.rb_x/2;
    Dtype center_2_y = rotateSquareRect_2.lt_y/2 + rotateSquareRect_2.rb_y/2;
    Dtype circumcircle_radius_2 = sqrt(  (rotateSquareRect_2.lt_x- rotateSquareRect_2.rb_x) * (rotateSquareRect_2.lt_x- rotateSquareRect_2.rb_x)
                                        +(rotateSquareRect_2.lt_y- rotateSquareRect_2.rb_y) * (rotateSquareRect_2.lt_y- rotateSquareRect_2.rb_y) )/2;

    Dtype center_dist = sqrt( (center_1_x -center_2_x) * (center_1_x -center_2_x) + (center_1_y -center_2_y) * (center_1_y -center_2_y) ); 

    if( center_dist >= circumcircle_radius_1 + circumcircle_radius_2 )
        return 0;
    else if( circumcircle_radius_1 >= center_dist + circumcircle_radius_2 ) 
        return getApproximateArea( rotateSquareRect_2 );
    else if( circumcircle_radius_2 >= center_dist + circumcircle_radius_1 )
        return getApproximateArea( rotateSquareRect_1 );
    

    Dtype area_CircumCircle_Intersection = getInterSectinArea_Circle( center_1_x, center_1_y, circumcircle_radius_1, 
                                                                      center_2_x, center_2_y, circumcircle_radius_2);

    return area_CircumCircle_Intersection*3/4;
}

template <typename Dtype>
const vector<bool> nms(vector< RotateBBox<Dtype> >& candidates,
    const Dtype overlap, const int top_N, const bool addScore) {
  vector<bool> mask(candidates.size(), false);

  if (mask.size() == 0) return mask;

  vector<bool> skip(candidates.size(), false);
  std::stable_sort(candidates.begin(), candidates.end(), RotateBBox<Dtype>::greater);

  vector<float> areas(candidates.size(), 0);
  for (int i = 0; i < candidates.size(); ++i) {
  	areas[i] = getApproximateArea(candidates[i]);
  }

  for (int count = 0, i = 0; count < top_N && i < mask.size(); ++i) {
    if (skip[i]) continue;

    mask[i] = true;
    ++count;

    // suppress the significantly covered bbox
    for (int j = i + 1; j < mask.size(); ++j) {
      if (skip[j]) continue;

      Dtype area_intersection = getApproximateInterSectionArea(candidates[i], candidates[j]);
      if ( area_intersection > 0 ){
        // compute overlap
        float o = area_intersection / std::min(areas[i],areas[j]);
        if (o > overlap) {
          skip[j] = true;

          if (addScore) {
          	candidates[i].score += candidates[j].score;
          }
        }
      }
    }
  }

  return mask;
}
template const vector<bool> nms  (vector< RotateBBox<float> >&  candidates,
    const float overlap, const int top_N, const bool addScore );
template const vector<bool> nms  (vector< RotateBBox<double> >&  candidates,
		const double overlap, const int top_N, const bool addScore );
/*
 *
 *  END of designed by Dukang
 */
 
 
template <typename Dtype>
Dtype  GetArea(const vector<Dtype>& bbox) {
	Dtype w = bbox[2] - bbox[0] + 1;
	Dtype h = bbox[3] - bbox[1] + 1;
	if (w <= 0 || h <= 0) return Dtype(0.0);

	return w * h;
}

template float GetArea(const vector<float>& bbox);
template double GetArea(const vector<double>& bbox);

template <typename Dtype>
Dtype GetArea(const Dtype x1, const Dtype y1, const Dtype x2, const Dtype y2)
{
	Dtype w = x2- x1 + 1;
	Dtype h = y2 - y1 + 1;
	if (w <= 0 || h <= 0) return Dtype(0.0);

	return w * h;
}

template float GetArea(const float x1, const float y1, const float x2, const float y2);
template double GetArea(const double x1, const double y1, const double x2, const double y2);


template <typename Dtype>
Dtype GetOverlap(const vector<Dtype>& bbox1, const vector<Dtype>& bbox2) {
	Dtype x1 = MAX(bbox1[0], bbox2[0]);
	Dtype y1 = MAX(bbox1[1], bbox2[1]);
	Dtype x2 = MIN(bbox1[2], bbox2[2]);
	Dtype y2 = MIN(bbox1[3], bbox2[3]);
	Dtype w = x2 - x1 + 1;
	Dtype h = y2 - y1 + 1;
	if (w <= 0 || h <= 0) return Dtype(0.0);

	Dtype intersection = w * h;
	Dtype area1 = GetArea(bbox1);
	Dtype area2 = GetArea(bbox2);
	Dtype u = area1 + area2 - intersection;

	return intersection / u;
}

template float GetOverlap(const vector<float>& bbox1, const vector<float>& bbox2);
template double GetOverlap(const vector<double>& bbox1, const vector<double>& bbox2);

template <typename Dtype>
Dtype GetOverlap(const Dtype x11, const Dtype y11, const Dtype x12, const Dtype y12,
		const Dtype x21, const Dtype y21, const Dtype x22, const Dtype y22,const OverlapType overlap_type)
{
	Dtype x1 = MAX(x11, x21);
	Dtype y1 = MAX(y11, y21);
	Dtype x2 = MIN(x12, x22);
	Dtype y2 = MIN(y12, y22);
	Dtype w = x2 - x1 + 1;
	Dtype h = y2 - y1 + 1;
	if (w <= 0 || h <= 0) return Dtype(0.0);

	Dtype intersection = w * h;
	Dtype area1 = GetArea(x11, y11, x12, y12);
	Dtype area2 = GetArea(x21, y21, x22, y22);
	Dtype u = 0;
	switch(overlap_type)
	{
		case OVERLAP_UNION:
		{
			u = area1 + area2 - intersection;
			break;
		}
		case OVERLAP_BOX1:
		{
			u = area1 ;
			break;
		}
		case OVERLAP_BOX2:
		{
			u = area2 ;
			break;
		}
		default:
                ;	//LOG(FATAL) << "Unknown type " << overlap_type;
	}

	return intersection / u;
}
template float GetOverlap(const float x11, const float y11, const float x12, const float y12,
		 const float x21, const float y21, const float x22, const float y22,const OverlapType overlap_type);
template double GetOverlap(const double x11, const double y11, const double x12, const double y12,
		 const double x21, const double y21, const double x22, const double y22,const OverlapType overlap_type);


template <typename Dtype>
Dtype GetNMSOverlap(const vector<Dtype>& bbox1, const vector<Dtype>& bbox2) {
	Dtype x1 = MAX(bbox1[0], bbox2[0]);
	Dtype y1 = MAX(bbox1[1], bbox2[1]);
	Dtype x2 = MIN(bbox1[2], bbox2[2]);
	Dtype y2 = MIN(bbox1[3], bbox2[3]);
	Dtype w = x2 - x1 + 1;
	Dtype h = y2 - y1 + 1;
	if (w <= 0 || h <= 0) return 0.0;

	Dtype area2 = GetArea(bbox2);
	return w * h / area2;
}

template float GetNMSOverlap(const vector<float>& bbox1, const vector<float>& bbox2);
template double GetNMSOverlap(const vector<double>& bbox1, const vector<double>& bbox2);

}


