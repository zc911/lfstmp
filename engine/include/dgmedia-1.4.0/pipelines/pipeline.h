/*============================================================================
 * File Name   : pipeline.h
 * Author      : tongliu@deepglint.com
 * Version     : 1.0.0.0
 * Copyright   : Copyright 2016 DeepGlint Inc.
 * Created on  : 2016年2月18日 下午9:33:41
 * Description : 
 * ==========================================================================*/
#ifndef PIPELINE_H_
#define PIPELINE_H_

#include <string>
#include <stdlib.h>
#include <stdio.h>

#include "models/error.h"
#include "utils/defines.h"

using namespace std;

namespace dgmedia
{

typedef void (*RuntimeErrorReached)(Error err);

typedef enum
{
	Unknown = 0,			//Undefined
	Created = 1,			//Pipeline has not been initialized yet
	Ready = 2,				//Pipeline has been initialized
	Running = 3,			//Pipeline is running now
} PipelineStatus;

class Pipeline
{
public:
	Pipeline();
	virtual ~Pipeline();
	virtual Error Initialize() = 0;
	virtual Error Run() = 0;
	virtual Error Stop() = 0;
	PipelineStatus GetStatus();

protected:
	PipelineStatus status_;
};

} /* namespace msrc */

#endif /* PIPELINE_H_ */
