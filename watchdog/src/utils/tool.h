/*
 * tool.h
 *
 *  Created on: Dec 23, 2015
 *      Author: chenzhen
 */

#ifndef TOOL_H_
#define TOOL_H_

#include <stdio.h>



void printHex(void *data, const int len, const char *title);

void generateKey(unsigned char *key, int len);

#endif /* TOOL_H_ */
