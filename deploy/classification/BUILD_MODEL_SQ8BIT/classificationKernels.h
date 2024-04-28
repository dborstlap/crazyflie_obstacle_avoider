#ifndef __CLASSIFICATIONKERNEL_H__
#define __CLASSIFICATIONKERNEL_H__

#include "AutoTilerLibTypes.h"
#include "Gap.h"
#include "classification.h"
#include "CNN_BasicKernels_SQ8.h"
#define _classification_L1_Memory_SIZE 46724
#define _classification_L2_Memory_SIZE 82300
#define _classification_L2_Memory_Dyn_SIZE 180448
extern char *classification_L1_Memory; /* Size given for generation: 46736 bytes, used: 46724 bytes */
extern char *classification_L2_Memory; /* Size used for generation (static): 82300 bytes */
extern char *classification_L2_Memory_Dyn; /* Size used for generation (dynamic): 180448 bytes */
extern void S3_Conv2d_16x1x3x3_MaxPool_2x2_Relu(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S6_Conv2d_32x16x3x3_MaxPool_2x2_Relu(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S9_Conv2d_64x32x3x3_MaxPool_2x2_Relu(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S12_Op_FULLY_CONNECTED_0_23_fusion(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern int classificationCNN_Construct();
extern int classificationCNN_Destruct();
extern int classificationCNN_Memory(int Which);
extern int classificationCNN(
		signed char * __restrict__ Input_1,
		signed char * __restrict__ Output_1);
#endif
