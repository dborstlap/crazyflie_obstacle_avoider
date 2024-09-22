#ifndef __GATENETKERNEL_H__
#define __GATENETKERNEL_H__

#include "AutoTilerLibTypes.h"
#include "Gap.h"
#include "gatenet.h"
#include "CNN_BasicKernels_SQ8.h"
#define _gatenet_L1_Memory_SIZE 45136
#define _gatenet_L2_Memory_SIZE 17796
#define _gatenet_L2_Memory_Dyn_SIZE 129600
extern char *gatenet_L1_Memory; /* Size given for generation: 46736 bytes, used: 45136 bytes */
extern char *gatenet_L2_Memory; /* Size used for generation (static): 17796 bytes */
extern char *gatenet_L2_Memory_Dyn; /* Size used for generation (dynamic): 129600 bytes */
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
extern void S9_Conv2d_16x32x3x3_MaxPool_2x2_Relu(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S12_Conv2d_16x16x3x3_MaxPool_2x2_Relu(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S15_Conv2d_16x16x3x3_MaxPool_2x2_Relu(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S18_Conv2d_16x16x3x3_Relu(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S21_Op_FULLY_CONNECTED_0_13_fusion(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern int gatenetCNN_Construct();
extern int gatenetCNN_Destruct();
extern int gatenetCNN_Memory(int Which);
extern int gatenetCNN(
		signed char * __restrict__ Input_1,
		signed char * __restrict__ Output_1);
#endif
