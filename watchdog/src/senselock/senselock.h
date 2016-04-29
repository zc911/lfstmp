#ifndef __SAMPLE_26_H__
#define __SAMPLE_26_H__

//By default, program compiled at IBM-compatible PC	has the address alignment issues which don't exist 
//in C51 environment. For convenience of data-handling,we use the compact-pack mode for storing the
//exchanging data-buffer used between PC and EliteIV device.For more ,pls refer to sample 17.

#include "../../include/watch_dog.h"
#include "../senselock/common.h"
#include "../senselock/crypt2.h"
#include "../senselock/sense4.h"

#ifdef _WIN32
#pragma pack(push, 1)
#else
#pragma pack(1)
#endif 

#define S4_DEEPGLINT_FOLDER "df01"

#define IO_PACKAGE_HEADER_SIZE 2 //IO package header length:tag+len
#define MAX_IO_DATA_SIZE	0xf8 // maximum IO data size
#define MAX_DATA_BLOCK_LEN (MAX_IO_DATA_SIZE - 9) // maximum user-specific data size

/* command group definition */
#define DES_ENC_ECB 0x00
#define DES_DEC_ECB 0x01
#define DES_ENC_CBC 0x02
#define DES_DEC_CBC 0x03
#define TDES_ENC_ECB 0x04
#define TDES_DEC_ECB 0x05
#define TDES_ENC_CBC 0x06
#define TDES_DEC_CBC 0x07

/* common error code. */
#define ERR_SUCCESS				0x00	// success
#define ERR_SES					0x80	// SES function execution error
#define ERR_INVALID_PARAMETER	0x81	// invalid parameter

/* user-specific data block for CBC mode*/
typedef struct _DATA_BLOCK
{
	unsigned char iv[8]; // initial vector
	unsigned char len;
	unsigned char buff[MAX_DATA_BLOCK_LEN];
} DATA_BLOCK;

/* IO package */
typedef struct _IO_PACKAGE
{	
	unsigned char tag;	
	unsigned char len;
	unsigned char buff[MAX_IO_DATA_SIZE];
} IO_PACKAGE;

#ifdef _WIN32
#pragma pack(pop)
#else
#pragma pack()
#endif


#endif //__SAMPLE_26_H__
