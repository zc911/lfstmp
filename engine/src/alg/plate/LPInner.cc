
#include "LPInner.hpp"


int copy_ndarray(NDArrayHandle hparam, NDArrayHandle hparamto, mx_float *pfBuffer, size_t buffersize) {
  int ret = 0;
  
  mx_uint out_dim = 0;
  const mx_uint *out_pdata = 0;
  ret = MXNDArrayGetShape(hparam, &out_dim, &out_pdata);
  size_t needsize = 1;
  for (int j = 0; j < out_dim; j++) {
    needsize *= out_pdata[j];
  }

  assert(buffersize >= needsize);
  
//  cout << 1 << endl;
  MXNDArraySyncCopyToCPU(hparam, pfBuffer, needsize);
//  cout << 2 << endl;
  MXNDArraySyncCopyFromCPU(hparamto, pfBuffer, needsize);

  return 0;
}


int getSize(NDArrayHandle hout) {
    mx_uint out_dim = 0;
    const mx_uint *out_shape = 0;
    int ret = MXNDArrayGetShape(hout, &out_dim, &out_shape);
    size_t needsize = 1;
    for (int j = 0; j < out_dim; j++) {
      needsize *= out_shape[j];
    }
    
    return needsize;
}


float calc_IOU(LPRectInfo &rect0, LPRectInfo &rect1) {
	float iou = 0.0f;
	float sz0 = rect0.fWidth * rect0.fHeight;
	float sz1 = rect1.fWidth * rect1.fHeight;
	
	float l0 = rect0.fCentX - rect0.fWidth / 2;
	float t0 = rect0.fCentY - rect0.fHeight / 2;
	float r0 = rect0.fCentX + rect0.fWidth / 2;
	float b0 = rect0.fCentY + rect0.fHeight / 2;

	float l1 = rect1.fCentX - rect1.fWidth / 2;
	float t1 = rect1.fCentY - rect1.fHeight / 2;
	float r1 = rect1.fCentX + rect1.fWidth / 2;
	float b1 = rect1.fCentY + rect1.fHeight / 2;

	float l01 = max(l0, l1);
	float t01 = max(t0, t1);
	float r01 = min(r0, r1);
	float b01 = min(b0, b1);

	if (r01 > l01 && b01 > t01) {
    iou = (r01 - l01 + 1) * (b01 - t01 + 1);
  }

  iou = iou / (sz0 + sz1 - iou + 0.001f);

  return iou;
}



int calc_overlap(LPRectInfo &rect0, LPRectInfo &rect1, float *pfOR0, float *pfOR1) {
	float iou = 0.0f;
	float sz0 = rect0.fWidth * rect0.fHeight;
	float sz1 = rect1.fWidth * rect1.fHeight;
	
	float l0 = rect0.fCentX - rect0.fWidth / 2;
	float t0 = rect0.fCentY - rect0.fHeight / 2;
	float r0 = rect0.fCentX + rect0.fWidth / 2;
	float b0 = rect0.fCentY + rect0.fHeight / 2;

	float l1 = rect1.fCentX - rect1.fWidth / 2;
	float t1 = rect1.fCentY - rect1.fHeight / 2;
	float r1 = rect1.fCentX + rect1.fWidth / 2;
	float b1 = rect1.fCentY + rect1.fHeight / 2;

	float l01 = max(l0, l1);
	float t01 = max(t0, t1);
	float r01 = min(r0, r1);
	float b01 = min(b0, b1);

	if (r01 > l01 && b01 > t01) {
    iou = (r01 - l01 + 1) * (b01 - t01 + 1);
  }

  *pfOR0 = iou / (sz0 + 0.001f);
  *pfOR1 = iou / (sz1 + 0.001f);

  return 0;
}


bool compare(const LPRectInfo a, const LPRectInfo b) {
	return a.fScore > b.fScore;
}


int group_bbs(vector<LPRectInfo> &lprects, vector<LPRectInfo> &group, float fiouThd) {
#define GBBS_MAXNUM 2048
	int rectnum = lprects.size();
	int num = 0;
	int adwMark[GBBS_MAXNUM];

	group.clear();
//  cout << "rectnum:" << rectnum << endl;
	sort(lprects.begin(), lprects.end(), compare);
//	for (int i = 0; i < rectnum; i++) {
//		cout << i << ":" << lprects[i].fScore << endl;
//	}
//	assert(rectnum <= GBBS_MAXNUM);

	vector<LPRectInfo> lprects_tmp0, lprects_tmp1;
	LPRectInfo lprtmp;
	for (int ri = 0; ri < rectnum; ri++) {
		lprects_tmp0.push_back(lprects[ri]);
	}

	while (1) {
	  if (num >= GBBS_MAXNUM) {
	    break;
	  }
		rectnum = lprects_tmp0.size();

		if (rectnum == 0) break;

		LPRectInfo lpr0 = lprects_tmp0[0];
		LPRectInfo lpr = lprects_tmp0[0];
		num = 1;
		adwMark[num - 1] = 0;
		lprects_tmp1.clear();
		for (int ri = 1; ri < rectnum; ri++) {
			lprtmp = lprects_tmp0[ri];
//			cout << lpr0.fScore << "," << lpr0.fCentX << "," << lpr0.fCentY << "," << lpr0.fHeight << "," << lpr0.fWidth << endl;
			float fratio = calc_IOU(lpr0, lprtmp);
//			cout << "====" << fratio << endl;
			if (fratio > fiouThd) {
				num++;
				lpr.fScore += lprtmp.fScore;
				lpr.fCentX += lprtmp.fCentX;
				lpr.fCentY += lprtmp.fCentY;
				lpr.fHeight += lprtmp.fHeight;
				lpr.fWidth += lprtmp.fWidth;
			}
			else {
				lprects_tmp1.push_back(lprtmp);
			}
		}

//		cout << "out:" << lprects_tmp1.size() << endl;

		lpr.fScore /= num;
		lpr.fCentX /= num;
		lpr.fCentY /= num;
		lpr.fHeight /= num;
		lpr.fWidth /= num;
		
//		cout << lpr.fScore << "," << lpr.fCentX << "," << lpr.fCentY << "," << lpr.fHeight << "," << lpr.fWidth << endl;
		if (group.size() < MAX_GP_NUMBER) {
			group.push_back(lpr);
		}
		else {
			break;
		}

		lprects_tmp0.clear();
		for (int ri = 0; ri < lprects_tmp1.size(); ri++) {
			lprects_tmp0.push_back(lprects_tmp1[ri]);
		}
	}
	
	sort(group.begin(), group.end(), compare);
	
//	cout << "group:" << group.size() << endl;

	return 0;
}



int group_bbs_overlap(vector<LPRectInfo> &lprects, vector<LPRectInfo> &group, float fiouThd) {
#define GBBSO_MAXNUM 2048
	int rectnum = lprects.size();
	int num = 0;
	int adwMark[GBBSO_MAXNUM];
	float afBB0[4], afBB1[4];

	group.clear();
//  cout << "rectnum:" << rectnum << endl;
	sort(lprects.begin(), lprects.end(), compare);
//	for (int i = 0; i < rectnum; i++) {
//		cout << i << ":" << lprects[i].fScore << endl;
//	}
//	assert(rectnum <= GBBSO_MAXNUM);

	vector<LPRectInfo> lprects_tmp0, lprects_tmp1;
	LPRectInfo lprtmp;
	for (int ri = 0; ri < rectnum; ri++) {
		lprects_tmp0.push_back(lprects[ri]);
	}

	while (1) {
	  if (num >= GBBSO_MAXNUM) {
	    break;
	  }
		rectnum = lprects_tmp0.size();

		if (rectnum == 0) break;

		LPRectInfo lpr0 = lprects_tmp0[0];
		LPRectInfo lpr = lprects_tmp0[0];
		num = 1;
		adwMark[num - 1] = 0;
		lprects_tmp1.clear();
		for (int ri = 1; ri < rectnum; ri++) {
			lprtmp = lprects_tmp0[ri];
//			cout << lpr0.fScore << "," << lpr0.fCentX << "," << lpr0.fCentY << "," << lpr0.fHeight << "," << lpr0.fWidth << endl;
      float fratio0 = 0.0f, fratio1 = 0.0f;
			calc_overlap(lpr0, lprtmp, &fratio0, &fratio1);
//			cout << "====" << fratio << endl;
			if (fratio0 > fiouThd || fratio1 > fiouThd) {
				num++;
				afBB0[0] = lpr.fCentX - lpr.fWidth / 2;
				afBB0[1] = lpr.fCentY - lpr.fHeight / 2;
				afBB0[2] = lpr.fCentX + lpr.fWidth / 2;
				afBB0[3] = lpr.fCentY + lpr.fHeight / 2;
				
				afBB1[0] = lprtmp.fCentX - lprtmp.fWidth / 2;
				afBB1[1] = lprtmp.fCentY - lprtmp.fHeight / 2;
				afBB1[2] = lprtmp.fCentX + lprtmp.fWidth / 2;
				afBB1[3] = lprtmp.fCentY + lprtmp.fHeight / 2;
				
				afBB0[0] = min(afBB0[0], afBB1[0]);
				afBB0[1] = min(afBB0[1], afBB1[1]);
				afBB0[2] = max(afBB0[2], afBB1[2]);
				afBB0[3] = max(afBB0[3], afBB1[3]);
				
				lpr.fScore += lprtmp.fScore;
				lpr.fHeight = afBB0[3] - afBB0[1] + 1;
				lpr.fWidth = afBB0[2] - afBB0[0] + 1;
				lpr.fCentX = afBB0[0] + lpr.fWidth / 2;
				lpr.fCentY = afBB0[1] + lpr.fHeight / 2;
				
			}
			else {
				lprects_tmp1.push_back(lprtmp);
			}
		}

//		cout << "out:" << lprects_tmp1.size() << endl;

		lpr.fScore /= num;
		
//		cout << lpr.fScore << "," << lpr.fCentX << "," << lpr.fCentY << "," << lpr.fHeight << "," << lpr.fWidth << endl;
		if (group.size() < MAX_GP_NUMBER) {
			group.push_back(lpr);
		}
		else {
			break;
		}

		lprects_tmp0.clear();
		for (int ri = 0; ri < lprects_tmp1.size(); ri++) {
			lprects_tmp0.push_back(lprects_tmp1[ri]);
		}
	}
	
	sort(group.begin(), group.end(), compare);
	
//	cout << "group:" << group.size() << endl;

	return 0;
}



void imgResizeAddBlack(uchar *patch, int s32W_src, int s32H_src,
													 uchar *tmpBuffer, uchar *result, 
													 int s32W_dst, int s32H_dst, int *pReal_w, int *pReal_h) {
	int rszH, rszW;
	
	memset(result, 0, s32W_dst * s32H_dst);

	if (s32W_src > s32W_dst || s32H_src > s32H_dst) {
		if (s32W_src * s32H_dst > s32H_src * s32W_dst) {
			rszW = s32W_dst;
			rszH = s32H_src * s32W_dst / s32W_src;
		}
		else {
			rszH = s32H_dst;
			rszW = s32W_src * s32H_dst / s32H_src;
		}
		
		imgResize(patch, s32W_src, s32H_src, tmpBuffer, rszW, rszH);

		for (int ri = 0; ri < rszH; ri++) {
			memcpy(result + ri * s32W_dst, tmpBuffer + ri * rszW, rszW);
		}
	}
	else {
		rszW = s32W_src;
		rszH = s32H_src;
		for (int ri = 0; ri < rszH; ri++) {
			memcpy(result + ri * s32W_dst, patch + ri * rszW, rszW);
		}
	}

	*pReal_w = rszW;
	*pReal_h = rszH;

}


void imgResize(uchar *patch, int s32W_src, int s32H_src, uchar *result, int s32W_dst, int s32H_dst) {
    int s32RI, s32CI;
    float srcX, srcY;
    float fSub_X, fSub_Y;
    int s32x, s32y;
    float fCov_X = 1.0 * (s32W_src - 1) / (s32W_dst - 1);
    float fCov_Y = 1.0 * (s32H_src - 1) / (s32H_dst - 1);
    float fTool = 0;
    
    for(s32RI = 0; s32RI < s32H_dst; s32RI++)
    {
        srcY = s32RI * fCov_Y;
        s32y = (int)srcY;
        fSub_Y = srcY - s32y;
        uchar *ptrPatch = patch + s32y * s32W_src;
        uchar *ptrResult = result + s32RI * s32W_dst;
        for(s32CI=0; s32CI < s32W_dst; s32CI++)
        {
            srcX = s32CI * fCov_X;
            s32x = (int)srcX;
            fSub_X = srcX - s32x;
            fTool = fSub_X * fSub_Y;
            if((s32x == s32W_src - 1) || (s32y == s32H_src - 1))
                ptrResult[s32CI] = ptrPatch[s32x];
            else
                ptrResult[s32CI] = (uchar)((1 - fSub_X - fSub_Y + fTool) * ptrPatch[s32x] +
                                           (fSub_Y - fTool) * ptrPatch[s32x + s32W_src] + (fSub_X - fTool) * ptrPatch[s32x + 1] +
                                           fTool * ptrPatch[s32x + s32W_src + 1] + 0.5);
        }
    }
}


void normalize_img_data(uchar *pubyImgData, int dwW, int dwH, int dwRatio)
{
	int dwPI, dwRI, dwCI;
	int adwHist[256];
	int adwMinInfo[2], adwMaxInfo[2];
	int dwMinV, dwMaxV;
	int dwTmp = 0, dwTmp2 = 0;
	int marginh = dwH / 10;
	int marginw = dwW / 10;
	int dwDataLen = dwW * dwH;

	memset(adwHist, 0, 256 * sizeof(int));
	for (dwRI = marginh; dwRI < dwH - marginh; dwRI++) {
		for (dwCI = marginw; dwCI < dwW - marginw; dwCI++) {
			dwPI = dwRI * dwW + dwCI;
			adwHist[pubyImgData[dwPI]]++;
		}
	}

	adwMinInfo[0] = 0;
	adwMinInfo[1] = 0;
	for (dwPI = 0; dwPI < 256; dwPI++)
	{
		adwMinInfo[0] += adwHist[dwPI];
		adwMinInfo[1] += adwHist[dwPI] * dwPI;
		if (adwMinInfo[0] * 100 > dwRatio * dwDataLen)
		{
			break;
		}
	}
	
	adwMaxInfo[0] = 0;
	adwMaxInfo[1] = 0;
	for (dwPI = 255; dwPI >= 0; dwPI--)
	{
		adwMaxInfo[0] += adwHist[dwPI];
		adwMaxInfo[1] += adwHist[dwPI] * dwPI;
		if (adwMaxInfo[0] * 100 > dwRatio * dwDataLen)
		{
			break;
		}
	}
	
	dwMinV = adwMinInfo[1] / adwMinInfo[0];
	dwMaxV = adwMaxInfo[1] / adwMaxInfo[0];
	dwTmp2 = dwMaxV - dwMinV + 1;
	for (dwPI = 0; dwPI < dwDataLen; dwPI++)
	{
		dwTmp = (pubyImgData[dwPI] - dwMinV) * 255 / dwTmp2;
		if (dwTmp < 0) dwTmp = 0;
		if (dwTmp > 255) dwTmp = 255;
		pubyImgData[dwPI] = dwTmp;
	}
}


int impSobelX_Abs_U8(uchar *pubyImg, int dwW, int dwH, uchar *pubySblX)
{
    int dwRI, dwCI;
    uchar *pubyRow1, *pubyRow2, *pubyRow3;
    uchar *pubyRow;
    short wVal;
    
    memset(pubySblX, 0, dwW * dwH);
    for (dwRI = 1; dwRI < dwH - 1; dwRI++) {
        pubyRow1 = pubyImg + (dwRI - 1) * dwW;
        pubyRow2 = pubyRow1 + dwW;
        pubyRow3 = pubyRow2 + dwW;
        pubyRow = pubySblX + dwRI * dwW;
        for (dwCI = 2; dwCI < dwW - 2; dwCI++) {
            if (pubyRow2[dwCI] * pubyRow2[dwCI-1] * pubyRow2[dwCI+1] * pubyRow2[dwCI-2] * pubyRow2[dwCI+2] == 0) {
                continue;
            }
            wVal = (pubyRow1[dwCI + 1] + pubyRow2[dwCI + 1] * 2 + pubyRow3[dwCI + 1]) - (pubyRow1[dwCI - 1] + pubyRow2[dwCI - 1] * 2 + pubyRow3[dwCI - 1]);
            wVal = abs(wVal) / 4;
            if (wVal > 255) {
                wVal = 255;
            }
            pubyRow[dwCI] = wVal;
        }
    }
    
    return 0;
}


void imgResizeAddBlack_f_bak(float *pfInputImg, int dwSrcW, int dwSrcH, float *pfDstImg, 
													 int dwDstW, int dwDstH, int *pdwRealW, int *pdwRealH)
{
	int rszH, rszW;
	
	cv::Mat srcImg(dwSrcH, dwSrcW, CV_32FC1, pfInputImg);
	cv::Mat dstImg(dwDstH, dwDstW, CV_32FC1, pfDstImg);

	if (dwSrcW > dwDstW || dwSrcH > dwDstH) {
		if (dwSrcW * dwDstH > dwSrcH * dwDstW) {
			rszW = dwDstW;
			rszH = dwSrcH * dwDstW / dwSrcW;
		}
		else {
			rszH = dwDstH;
			rszW = dwSrcW * dwDstH / dwSrcH;
		}
		
		cv::Mat tmpImg;
		cv::resize(srcImg, tmpImg, cv::Size(rszW, rszH), 0, 0, CV_INTER_LINEAR);
    cv::copyMakeBorder(tmpImg, dstImg, 0, dwDstH - rszH, 0, dwDstW - rszW, cv::BORDER_CONSTANT, 0);
	}
	else {
		rszW = dwSrcW;
		rszH = dwSrcH;
    cv::copyMakeBorder(srcImg, dstImg, 0, dwDstH - rszH, 0, dwDstW - rszW, cv::BORDER_CONSTANT, 0);
	}

	*pdwRealW = rszW;
	*pdwRealH = rszH;

#if LPDR_DBG&0
  cv::Mat blackImg(dwDstH, dwDstW, CV_32FC1, pfDstImg);
  cv::namedWindow("addblack", 0);
  cv::imshow("addblack", blackImg);
  cv::waitKey(10);
#endif

  return;
}


void imgResizeAddBlack_f(float *pfInputImg, int dwSrcW, int dwSrcH, float *pfDstImg, 
													 int dwDstW, int dwDstH, int *pdwRealW, int *pdwRealH)
{
	int rszH, rszW;
	
	cv::Mat srcImg(dwSrcH, dwSrcW, CV_32FC1, pfInputImg);
	cv::Mat dstImg(dwDstH, dwDstW, CV_32FC1, pfDstImg);

	if (dwSrcW > dwDstW || dwSrcH > dwDstH) {
		if (dwSrcW * dwDstH > dwSrcH * dwDstW) {
			rszW = dwDstW;
			rszH = dwSrcH * dwDstW / dwSrcW;
		}
		else {
			rszH = dwDstH;
			rszW = dwSrcW * dwDstH / dwSrcH;
		}
		
		cv::Mat tmpImg;
		cv::resize(srcImg, tmpImg, cv::Size(rszW, rszH), 0, 0, CV_INTER_LINEAR);
    float *pfTmpData = (float*)tmpImg.data;
		for (int ri = 0; ri < rszH; ri++) {
			memcpy(pfDstImg + ri * dwDstW, pfTmpData + ri * rszW, rszW * sizeof(float));
		}
	}
	else {
		rszW = dwSrcW;
		rszH = dwSrcH;
		for (int ri = 0; ri < rszH; ri++) {
			memcpy(pfDstImg + ri * dwDstW, pfInputImg + ri * rszW, rszW * sizeof(float));
		}
	}

	*pdwRealW = rszW;
	*pdwRealH = rszH;

#if LPDR_DBG&0
  cv::Mat blackImg(dwDstH, dwDstW, CV_32FC1, pfDstImg);
  cv::namedWindow("addblack", 0);
  cv::imshow("addblack", blackImg);
  cv::waitKey(10);
#endif

  return;
}

void imgResizeAddBlack_f_bak2(float *pfInputImg, int dwSrcW, int dwSrcH, float *pfDstImg, 
													 int dwDstW, int dwDstH, int *pdwRealW, int *pdwRealH)
{
	int rszH, rszW;

	if (dwSrcW > dwDstW || dwSrcH > dwDstH) {
		if (dwSrcW * dwDstH > dwSrcH * dwDstW) {
			rszW = dwDstW;
			rszH = dwSrcH * dwDstW / dwSrcW;
		}
		else {
			rszH = dwDstH;
			rszW = dwSrcW * dwDstH / dwSrcH;
		}
	}
	else {
		rszW = dwSrcW;
		rszH = dwSrcH;
	}

  int dwRI, dwCI;
  int dwSrcRI_Offset, dwDstRI_Offset, dwSrcOft, dwDstOft;
  for (dwRI = 0; dwRI < rszH; dwRI++) {
    dwSrcRI_Offset = (dwRI * dwSrcH / rszH) * dwSrcW;
    dwDstRI_Offset = dwRI * dwDstW;
    for (dwCI = 0; dwCI < rszW; dwCI++) {
      dwSrcOft = dwCI * dwSrcW / rszW + dwSrcRI_Offset;
      dwDstOft = dwCI + dwDstRI_Offset;
      pfDstImg[dwDstOft] = pfInputImg[dwSrcOft];
    }
  }

	*pdwRealW = rszW;
	*pdwRealH = rszH;

#if LPDR_DBG
  cv::Mat blackImg(dwDstH, dwDstW, CV_32FC1, pfDstImg);
  cv::namedWindow("addblack", 0);
  cv::imshow("addblack", blackImg);
  cv::waitKey(0);
#endif

  return;
}

int doNormContrastBB_f(float *pfImage, int dwH, int dwW, LPRect bb)
{
  int dwRI, dwCI;
  float fMin, fMax, fVal;
  int dwX0 = bb.dwX0;
  int dwY0 = bb.dwY0;
  int dwX1 = bb.dwX1;
  int dwY1 = bb.dwY1;
  float *pfRow;
  
  fMin = 1.0f;
  fMax = 0.0f;
  for (dwRI = dwY0; dwRI < dwY1; dwRI++)
  {
    pfRow = pfImage + dwRI * dwW;
    for (dwCI = dwX0; dwCI < dwX1; dwCI++)
    {
      if (pfRow[dwCI] < fMin) fMin = pfRow[dwCI];
      if (pfRow[dwCI] > fMax) fMax = pfRow[dwCI];
    }
  }
  
  if (fMax < fMin + 0.001f) fMax = fMin + 0.001f;
  
  for (dwRI = 0; dwRI < dwW * dwH; dwRI++)
  {
    fVal = (pfImage[dwRI] - fMin) / (fMax - fMin);
    if (fVal < 0.f) fVal = 0.f;
    if (fVal > 1.f) fVal = 1.f;
    pfImage[dwRI] = fVal;
  }
  
  return 0;
}


int calcNewMarginBB(int dwImgH, int dwImgW, LPRect *pstBB, int adwMRatioXY[2])
{
  LPRect bb_old = *pstBB;
  int dwMarginX = (bb_old.dwX1 - bb_old.dwX0 + 1) / adwMRatioXY[0];
  int dwMarginY = (bb_old.dwY1 - bb_old.dwY0 + 1) / adwMRatioXY[1];
  
  bb_old.dwX0 = max(0, bb_old.dwX0-dwMarginX);
  bb_old.dwX1 = min(dwImgW-1, bb_old.dwX1+dwMarginX);
  bb_old.dwY0 = max(0, bb_old.dwY0-dwMarginY);
  bb_old.dwY1 = min(dwImgH-1, bb_old.dwY1+dwMarginY);
  
  *pstBB = bb_old;

  return 0;
}


int doRectify_f(float *pfImage0, float *pfImage1, int dwW, int dwH, float fAngle_old, int adwPolygonXY[8], float *pfAngle_new)
{
  float afVecs[4], afVec2[2];
  float afPnts[4*3];
  
  afVecs[0] = adwPolygonXY[1*2+0] - adwPolygonXY[0*2+0];
  afVecs[1] = adwPolygonXY[1*2+1] - adwPolygonXY[0*2+1];
  afVecs[2] = adwPolygonXY[2*2+0] - adwPolygonXY[3*2+0];
  afVecs[3] = adwPolygonXY[2*2+1] - adwPolygonXY[3*2+1];
  afVec2[0] = (afVecs[0] + afVecs[2]) / 2;
  afVec2[1] = (afVecs[1] + afVecs[3]) / 2;
  
  float fAngle = atan2(afVec2[1], afVec2[0]);
  fAngle = fAngle * 180 / 3.14159f;
  
//  cout << "doRectify_f 0\n";
  cv::Mat matRotate(2, 3, CV_32FC1);
  matRotate = getRotationMatrix2D(cv::Point(dwW/2, dwH/2), fAngle, 1.0);
  matRotate.convertTo(matRotate, CV_32FC1);
//  cout << "doRectify_f 1\n";
  afPnts[0*3+0] = adwPolygonXY[0*2+0]; afPnts[0*3+1] = adwPolygonXY[0*2+1]; afPnts[0*3+2] = 1.0f;
  afPnts[1*3+0] = adwPolygonXY[1*2+0]; afPnts[1*3+1] = adwPolygonXY[1*2+1]; afPnts[1*3+2] = 1.0f;
  afPnts[2*3+0] = adwPolygonXY[2*2+0]; afPnts[2*3+1] = adwPolygonXY[2*2+1]; afPnts[2*3+2] = 1.0f;
  afPnts[3*3+0] = adwPolygonXY[3*2+0]; afPnts[3*3+1] = adwPolygonXY[3*2+1]; afPnts[3*3+2] = 1.0f;
  
  cv::Mat matPnts(4, 3, CV_32FC1, afPnts);
  cv::Mat newPnts(4, 2, CV_32FC1);
  cv::Mat matRotateT(2, 3, CV_32FC1);
  matRotateT = matRotate.t();
  
  newPnts = matPnts * matRotateT;
  
  cv::Mat matImageSrc(dwH, dwW, CV_32FC1, pfImage0);
  cv::Mat matImageDst(dwH, dwW, CV_32FC1, pfImage1);
  
  float fAngle_new = fAngle_old + fAngle;
  matRotate = getRotationMatrix2D(cv::Point(dwW/2, dwH/2), fAngle_new, 1.0);
//  cout << "doRectify_f 3\n";
  warpAffine(matImageSrc, matImageDst, matRotate, matImageDst.size());
//  cout << "doRectify_f 4\n";
  *pfAngle_new = fAngle_new;
  
  float *pfNewPnts = (float*)newPnts.data;
  adwPolygonXY[0*2+0] = pfNewPnts[0*2+0]; adwPolygonXY[0*2+1] = pfNewPnts[0*2+1];
  adwPolygonXY[1*2+0] = pfNewPnts[1*2+0]; adwPolygonXY[1*2+1] = pfNewPnts[1*2+1];
  adwPolygonXY[2*2+0] = pfNewPnts[2*2+0]; adwPolygonXY[2*2+1] = pfNewPnts[2*2+1];
  adwPolygonXY[3*2+0] = pfNewPnts[3*2+0]; adwPolygonXY[3*2+1] = pfNewPnts[3*2+1];
  
  return 0;
}


int doRotate_f(float *pfImage, int dwW, int dwH, float fAngle)
{
  cv::Mat matRotate(2, 3, CV_32FC1);
  cv::Mat matImageSrc(dwH, dwW, CV_32FC1, pfImage);
  cv::Mat matImageDst(dwH, dwW, CV_32FC1);
  matRotate = getRotationMatrix2D(cv::Point(dwW/2, dwH/2), fAngle, 1.0);
  warpAffine(matImageSrc, matImageDst, matRotate, matImageDst.size());
  matImageDst.copyTo(matImageSrc);
  
  return 0;
}


int doRotate_8UC3(uchar *pubyImage, int dwW, int dwH, float fAngle)
{
  cv::Mat matRotate(2, 3, CV_8UC3);
  cv::Mat matImageSrc(dwH, dwW, CV_8UC3, pubyImage);
  cv::Mat matImageDst(dwH, dwW, CV_8UC3);
  matRotate = getRotationMatrix2D(cv::Point(dwW/2, dwH/2), fAngle, 1.0);
  warpAffine(matImageSrc, matImageDst, matRotate, matImageDst.size());
  matImageDst.copyTo(matImageSrc);
  
  return 0;
}


int getBinThresholdIterByHist_uchar(uchar *pubyData, int dwLen)
{
  int dwPI;
  int adwMeans[2] = {0, 0};
  int dwThreshold0, dwThreshold1;
  int adwHist[256];
  
  memset(adwHist, 0, 256 * sizeof(int));
  for (dwPI = 0; dwPI < dwLen; dwPI++)
  {
    adwHist[pubyData[dwPI]]++;
  }
  dwThreshold0 = getMeanByHist(adwHist, 256);
  for (dwPI = 0; dwPI < 16; dwPI++)
  {
    adwMeans[0] = getMeanByHist(adwHist, dwThreshold0);
    adwMeans[1] = getMeanByHist(adwHist + dwThreshold0, 256 - dwThreshold0) + dwThreshold0;
    dwThreshold1 = (adwMeans[0] + adwMeans[1]) / 2;
    if (abs(dwThreshold1 - dwThreshold0) < 2.0) break;
  }

  return dwThreshold1;
}


int getMeanByHist(int *pdwHist, int dwLen)
{
  int dwMean = 0;
  int dwNum = 1;
  for (int dwPI = 0; dwPI < dwLen; dwPI++)
  {
    dwMean += pdwHist[dwPI] * dwPI;
    dwNum += pdwHist[dwPI];
  }
  dwMean /= dwNum;
  
  return dwMean;
}


int cvtRGB2HSV_U8(uchar ubyR, uchar ubyG, uchar ubyB, float *pfH, float *pfS, float *pfV)
{
  //H: 0~360, S: 0~1, V: 0~1
  int dwR = ubyR, dwG = ubyG, dwB = ubyB;
  int dwMax, dwMin;
  float fH = 0.f, fS = 0.f, fV = 0.f;
  
  dwMax = max(dwR, max(dwG, dwB));
  dwMin = min(dwR, min(dwG, dwB));
  
  if (dwMax == dwMin)
  {
    fH = 0.f;
  }
  else if (dwMax == dwR && dwG >= dwB)
  {
    fH = 60.f * (dwG - dwB) / (dwMax - dwMin);
  }
  else if (dwMax == dwR && dwG < dwB)
  {
    fH = 60.f * (dwG - dwB) / (dwMax - dwMin) + 360.f;
  }
  else if (dwMax == dwG)
  {
    fH = 60.f * (dwB - dwR) / (dwMax - dwMin) + 120.f;
  }
  else if (dwMax == dwB)
  {
    fH = 60.f * (dwR - dwG) / (dwMax - dwMin) + 240.f;
  }
  
  if (dwMax == 0)
  {
    fS = 0.f;
  }
  else
  {
    fS = 1.0f - dwMin * 1.0f / dwMax;
  }
  
  fV = dwMax / 255.0f;

  *pfH = fH;
  *pfS = fS;
  *pfV = fV;

  return 0;
}









