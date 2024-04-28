#include "classificationKernels.h"
L1_CL_MEM AT_L1_POINTER classification_L1_Memory;
L2_MEM AT_L2_POINTER classification_L2_Memory;
L2_MEM AT_L2_POINTER classification_L2_Memory_Dyn;
AT_HYPERRAM_POINTER classification_L3_Memory;
static AT_HYPERRAM_T HyperRam;
static AT_HYPERFLASH_FS_T HyperFlash;
void S3_Conv2d_16x1x3x3_MaxPool_2x2_Relu(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 37344 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT _DmaW_Evt1, *DmaW_Evt1 = &_DmaW_Evt1;
	AT_L2_EVENT _DmaR_Evt1, *DmaR_Evt1 = &_DmaR_Evt1;
	AT_L2_EVENT _DmaR_Evt2, *DmaR_Evt2 = &_DmaR_Evt2;
	AT_L2_EVENT _DmaR_Evt3, *DmaR_Evt3 = &_DmaR_Evt3;
	AT_L2_EVENT _DmaR_Evt4, *DmaR_Evt4 = &_DmaR_Evt4;
	AT_L2_EVENT _DmaR_Evt5, *DmaR_Evt5 = &_DmaR_Evt5;
	AT_L2_EVENT _DmaR_Evt6, *DmaR_Evt6 = &_DmaR_Evt6;
	AT_HYPERRAM_CL_EVENT _UchanHR1, *UchanHR1 = &_UchanHR1;
	KerSetBias_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;
	KerConv_SQ8_T S_KerArg1, *KerArg1 = &S_KerArg1;
	KerConvLinReduct_SQ8_T S_KerArg2, *KerArg2 = &S_KerArg2;
	KerPool_SQ8_T S_KerArg3, *KerArg3 = &S_KerArg3;

	/* Iteration space related variables */
	int D1Ind, D1Ind_Last;
	int T0Ind, T0Ind_Total=0, T0Ind_Last, T0Ind_NextLast;
	int D0Ind, D0Ind_Last;
	/* User kernel arguments related variables */
	unsigned int _P_Out, _C_Out;
	unsigned int _SPP_Out, _SP_Out, _SC_Out;
	unsigned int _LPP_Out, _LP_Out, _LC_Out;
	unsigned int _N_In;
	unsigned int _SN_In;
	unsigned int _LN_In;
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D1 Dim: Init: 16, Tiled: 1][Tile0 Dim: 162][D0 Dim: Init: 1, Tiled: 1]
	Ker Arg: Out, Tiled Space: Tile0
		Min Pipe Depth: -2, Max Pipe Depth: 0
		KerArgItSpace: 162 logical tiles, 162 physical tiles
			Total Size: 316224 [D1, [0 x 316224, 316224]][Tile0, 162:[122x1, 160:122x1, 122x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 316224, 316224]][Tile0, 162:[122x1, 160:122x1, 122x1], 1]
		Tile0: [0, 1952, 1], Tile1: [1, 1952, 1], Tile2; [2, 1952, 1]
	Ker Arg: Bias, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 64 [D1, [0 x 64, 64]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 64, 64]]
		Tile0: [0, 64, 64], Tile1: [0, 64, 64], Tile2; [0, 64, 64]
	Ker Arg: Scale, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 16 [D1, [0 x 16, 16]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 16, 16]]
		Tile0: [0, 16, 16], Tile1: [0, 16, 16], Tile2; [0, 16, 16]
	Ker Arg: ScaleN, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 16 [D1, [0 x 16, 16]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 16, 16]]
		Tile0: [0, 16, 16], Tile1: [0, 16, 16], Tile2; [0, 16, 16]
	Ker Arg: Filter, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 144 [D1, [0 x 144, 144]][D0, [0 x 144, 144]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 144, 144]][D0, [0 x 144, 144]]
		Tile0: [0, 144, 144], Tile1: [0, 144, 144], Tile2; [0, 144, 144]
	Ker Arg: In, Tiled Space: Tile0
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 162 logical tiles, 162 physical tiles
			Total Size: 79056 [D0, [0 x 79056, 79056]][Tile0, 162:[244x3, 160:244x4, 244x3], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 162:[244x2], 1][D0, [0 x 79056, 79056]]
		Tile0: [0, 732, 3], Tile1: [1, 976, 4], Tile2; [3, 976, 4]
	Ker Arg: ConvOut, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 162 logical tiles, 1 physical tiles
			Total Size: 5059584 [D1, [0 x 5059584, 5059584]][Tile0, 162:[244x2, 160:244x2, 244x2], 4]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 5059584, 5059584]][Tile0, 162:[244x2, 160:244x2, 244x2], 4]
		Tile0: [0, 31232, 8], Tile1: [0, 31232, 8], Tile2; [0, 31232, 8]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 162 logical tiles, 1 physical tiles
			Total Size: 13 [Tile0, 162:[1x13, 160:1x13, 1x13], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 162:[1x13, 160:1x13, 1x13], 1]
		Tile0: [0, 13, 13], Tile1: [0, 13, 13], Tile2; [0, 13, 13]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->Out = (int * __restrict__) (classification_L1_Memory+6096);
	KerArg0->W = (unsigned short int) (2);
	KerArg0->H = (unsigned short int) (244);
	KerArg0->Feat = (unsigned short int) (16);
	KerArg0->Bias = (void * __restrict__) (classification_L1_Memory+1952);
	KerArg1->H = (unsigned short int) (244);
	KerArg1->UsedH = (unsigned short int) (244);
	KerArg1->InFeatures = (unsigned short int) (1);
	KerArg1->OutFeatures = (unsigned short int) (16);
	KerArg1->TotalInFeatures = (unsigned short int) (1);
	KerArg1->Filter = (signed char * __restrict__) (classification_L1_Memory+2048);
	KerArg1->Out = (int * __restrict__) (classification_L1_Memory+6096);
	KerArg2->In = (int *__restrict__) (classification_L1_Memory+6096);
	KerArg2->Out = (void *__restrict__) (classification_L1_Memory+6096);
	KerArg2->Feat = (unsigned short int) (16);
	KerArg2->W = (unsigned short int) (2);
	KerArg2->H = (unsigned short int) (244);
	KerArg2->Scale = (unsigned char *__restrict__) (classification_L1_Memory+2016);
	KerArg2->ScaleN = (unsigned char *__restrict__) (classification_L1_Memory+2032);
	KerArg2->Infos = (signed char *__restrict__) (classification_L1_Memory+37328);
	KerArg3->In = (signed char * __restrict__) (classification_L1_Memory+6096);
	KerArg3->W = (unsigned short int) (2);
	KerArg3->UsedW = (unsigned short int) (2);
	KerArg3->H = (unsigned short int) (244);
	KerArg3->UsedH = (unsigned short int) (244);
	KerArg3->Feat = (unsigned short int) (16);
	KerArg3->Pad = (v4s) 0;
	KerArg3->PoolMax = (unsigned char) (1);
	KerArg3->DoScale = (unsigned char) (0);
	KerArg3->Infos = (signed char * __restrict__) (classification_L1_Memory+37328);
	/*================================= Read Tiles Prolog ===============================*/
	_C_Out=0; _SC_Out=1952; _LC_Out=1;
	_SPP_Out=0; _SP_Out=0;
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+1952), 64, 0, DmaR_Evt1);
	AT_L2_WAIT(0, DmaR_Evt1); /* Wait previous DMA read Bias */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Scale+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+2016), 16, 0, DmaR_Evt2);
	AT_L2_WAIT(0, DmaR_Evt2); /* Wait previous DMA read Scale */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) ScaleN+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+2032), 16, 0, DmaR_Evt3);
	AT_L2_WAIT(0, DmaR_Evt3); /* Wait previous DMA read ScaleN */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Filter+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+2048), 144, 0, DmaR_Evt4);
	AT_L2_WAIT(0, DmaR_Evt4); /* Wait previous DMA read Filter */
	AT_L2_COPY2D(0, ((AT_L2_EXT_ADDR_TYPE) In+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+0+0), 732, 324, 3, 0, DmaR_Evt5);
	_N_In=0;
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+37328), 13, 0, DmaR_Evt6);
	AT_L2_WAIT(0, DmaR_Evt6); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on D1 */
		int D1Ind_Last = 1;
		for (T0Ind=0; T0Ind<162; T0Ind++, T0Ind_Total++) { /* Iteration on Tile0 */
			int T0Ind_Last = (T0Ind==161), T0Ind_NextLast = ((T0Ind+1)==161);
			/*================================= Prepare Tiles ===================================*/
			_SN_In = 0;
			if (!(T0Ind_Last)) {
				_N_In = _N_In + (2-(1*(T0Ind==0))); _LN_In = ((T0Ind_NextLast)?3:4); _SN_In = (244*_LN_In); 
			} else if (!(1)) {
				_N_In = _N_In + (-321); _LN_In = (3); _SN_In = (244*_LN_In); 
			}
			/*============================= End Prepare Tiles ===================================*/
			/*================================= Read Tiles ======================================*/
			AT_L2_WAIT(0, DmaR_Evt5); /* Wait previous DMA read In */
			if (_SN_In) {
				AT_L2_COPY2D(0, ((AT_L2_EXT_ADDR_TYPE) In+_N_In), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+0+976*((T0Ind_Total+1)%2)),
						_SN_In, 324, _LN_In, 0, DmaR_Evt5);
			}
			/*============================= End Read Tiles ======================================*/
			/*====================== Call Kernel LOC_D0_PROLOG =========================*/
			KerArg0->NormBias = (unsigned char) (((char *)(classification_L1_Memory+37328))[8]);
			AT_FORK(gap_ncore(), (void *) KerParSetBiasB32_SQ8, (void *) KerArg0);
			__CALL(KerParSetBiasB32_SQ8, KerArg0);
			{ /* Single iteration on D0 */
				int D0Ind_Last = 1;
				/*====================== Call Kernel LOC_D0 =========================*/
				KerArg1->In = (signed char * __restrict__) (classification_L1_Memory+0+976*((T0Ind_Total)%2));
				KerArg1->W = (unsigned short int) (4-1*(T0Ind==0)-1*(T0Ind_Last));
				KerArg1->UsedW = (unsigned short int) (4-1*(T0Ind==0)-1*(T0Ind_Last));
				KerArg1->Pad = (v4s) ((v4s){1*(T0Ind==0),1*(T0Ind_Last),1,1});
				AT_FORK(gap_ncore(), (void *) KerParConv3x3Stride1_SQ8, (void *) KerArg1);
				__CALL(KerParConv3x3Stride1_SQ8, KerArg1);
			} /* End iteration on D0 */
			/*====================== Call Kernel LOC_D0_EPILOG =========================*/
			AT_FORK(gap_ncore(), (void *) KerParReductIO_CC_SQ8, (void *) KerArg2);
			__CALL(KerParReductIO_CC_SQ8, KerArg2);
			KerArg3->Out = (signed char * __restrict__) (classification_L1_Memory+2192+1952*((T0Ind_Total)%2));
			AT_FORK(gap_ncore(), (void *) KerParPool2x2Stride2_ReLU_SQ8, (void *) KerArg3);
			__CALL(KerParPool2x2Stride2_ReLU_SQ8, KerArg3);
			/*================================= Write Tiles =====================================*/
			if (_SP_Out) AT_L2_WAIT(0, DmaW_Evt1); /* Wait previous DMA write Out */
			if (_SPP_Out) AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR1); /* Wait previous uDMA write Out */
			if (_SP_Out) AT_HYPERRAM_CL_COPY2D(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) Out+_P_Out), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn+0+1952*((T0Ind_Total+-1)%2)),
						_SP_Out, 162, _LP_Out, 1, UchanHR1);
			AT_L2_COPY(0, ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn+0+1952*((T0Ind_Total)%2)), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+2192+1952*((T0Ind_Total)%2)),
					_SC_Out, 1, DmaW_Evt1);
			/*============================= End Write Tiles =====================================*/
			/*================================= Update Arg Pipeline =============================*/
			_SPP_Out = _SP_Out;_LPP_Out = _LP_Out;
			_P_Out = _C_Out;_SP_Out = _SC_Out;_LP_Out = _LC_Out;
			/*============================= End Update Arg Pipeline =============================*/
			/*================================= Prepare Tiles ===================================*/
			_SC_Out = 0;
			if (!(T0Ind_Last)) {
				_C_Out = _C_Out + (1); _LC_Out = (1); _SC_Out = (1952*_LC_Out); 
			}
			/*============================= End Prepare Tiles ===================================*/
		} /* End iteration on Tile0 */
	} /* End iteration on D1 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_WAIT(0, DmaW_Evt1); /* Wait previous DMA write Out */
	if (_SPP_Out) AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR1); /* Wait previous uDMA write Out */
	AT_HYPERRAM_CL_COPY2D(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) Out+_P_Out), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn+0+1952*((T0Ind_Total+-1)%2)), _SP_Out, 162, _LP_Out, 1, UchanHR1);
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR1); /* Wait current uDMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S6_Conv2d_32x16x3x3_MaxPool_2x2_Relu(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 43856 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT _DmaW_Evt1, *DmaW_Evt1 = &_DmaW_Evt1;
	AT_L2_EVENT _DmaR_Evt1, *DmaR_Evt1 = &_DmaR_Evt1;
	AT_L2_EVENT _DmaR_Evt2, *DmaR_Evt2 = &_DmaR_Evt2;
	AT_L2_EVENT _DmaR_Evt3, *DmaR_Evt3 = &_DmaR_Evt3;
	AT_L2_EVENT _DmaR_Evt4, *DmaR_Evt4 = &_DmaR_Evt4;
	AT_L2_EVENT _DmaR_Evt5, *DmaR_Evt5 = &_DmaR_Evt5;
	AT_L2_EVENT _DmaR_Evt6, *DmaR_Evt6 = &_DmaR_Evt6;
	AT_HYPERRAM_CL_EVENT _UchanHR1, *UchanHR1 = &_UchanHR1;
	KerSetBias_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;
	KerConv_SQ8_T S_KerArg1, *KerArg1 = &S_KerArg1;
	KerConvLinReduct_SQ8_T S_KerArg2, *KerArg2 = &S_KerArg2;
	KerPool_SQ8_T S_KerArg3, *KerArg3 = &S_KerArg3;

	/* Iteration space related variables */
	int D1Ind, D1Ind_Last;
	int T0Ind, T0Ind_Total=0, T0Ind_Last, T0Ind_NextLast, T0Ind_NextNextLast;
	int D0Ind, D0Ind_Total=0, D0Ind_Last, D0Ind_NextLast, D0Ind_NextNextLast;
	/* User kernel arguments related variables */
	unsigned int _C_Out;
	unsigned int _SP_Out, _SC_Out;
	unsigned int _LP_Out, _LC_Out;
	unsigned int _NN_In;
	unsigned int _SN_In, _SNN_In;
	unsigned int _LN_In, _LNN_In;
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D1 Dim: Init: 32, Tiled: 1][Tile0 Dim: 81][D0 Dim: Init: 16, Tiled: 4]
	Ker Arg: Out, Tiled Space: Tile0
		Min Pipe Depth: -1, Max Pipe Depth: 0
		KerArgItSpace: 81 logical tiles, 81 physical tiles
			Total Size: 158112 [D1, [0 x 158112, 158112]][Tile0, 81:[61x1, 79:61x1, 61x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 158112, 158112]][Tile0, 81:[61x1, 79:61x1, 61x1], 1]
		Tile0: [0, 1952, 1], Tile1: [1, 1952, 1], Tile2; [2, 1952, 1]
	Ker Arg: Bias, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 128 [D1, [0 x 128, 128]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 128, 128]]
		Tile0: [0, 128, 128], Tile1: [0, 128, 128], Tile2; [0, 128, 128]
	Ker Arg: Scale, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 32 [D1, [0 x 32, 32]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 32, 32]]
		Tile0: [0, 32, 32], Tile1: [0, 32, 32], Tile2; [0, 32, 32]
	Ker Arg: ScaleN, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 32 [D1, [0 x 32, 32]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 32, 32]]
		Tile0: [0, 32, 32], Tile1: [0, 32, 32], Tile2; [0, 32, 32]
	Ker Arg: Filter, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 4 logical tiles, 1 physical tiles
			Total Size: 4608 [D1, [0 x 4608, 4608]][D0, [3 x 1152, 1152]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 4608, 4608]][D0, [3 x 1152, 1152]]
		Tile0: [0, 4608, 4608], Tile1: [0, 4608, 4608], Tile2; [0, 4608, 4608]
	Ker Arg: In, Tiled Space: Tile0
		Min Pipe Depth: 0, Max Pipe Depth: 2
		KerArgItSpace: 324 logical tiles, 324 physical tiles
			Total Size: 316224 [D0, [3 x 79056, 79056]][Tile0, 81:[122x3, 79:122x4, 122x3], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 81:[122x3, 2:122x4, 122x3], 1][D0, [3 x 79056, 79056]]
		Tile0: [0, 1464, 3], Tile1: [79056, 1464, 3], Tile2; [158112, 1464, 3]
	Ker Arg: ConvOut, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 81 logical tiles, 1 physical tiles
			Total Size: 2529792 [D1, [0 x 2529792, 2529792]][Tile0, 81:[122x2, 79:122x2, 122x2], 4]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 2529792, 2529792]][Tile0, 81:[122x2, 79:122x2, 122x2], 4]
		Tile0: [0, 31232, 8], Tile1: [0, 31232, 8], Tile2; [0, 31232, 8]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 81 logical tiles, 1 physical tiles
			Total Size: 13 [Tile0, 81:[1x13, 79:1x13, 1x13], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 81:[1x13, 79:1x13, 1x13], 1]
		Tile0: [0, 13, 13], Tile1: [0, 13, 13], Tile2; [0, 13, 13]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->Out = (int * __restrict__) (classification_L1_Memory+12608);
	KerArg0->W = (unsigned short int) (2);
	KerArg0->H = (unsigned short int) (122);
	KerArg0->Feat = (unsigned short int) (32);
	KerArg0->Bias = (void * __restrict__) (classification_L1_Memory+3904);
	KerArg1->H = (unsigned short int) (122);
	KerArg1->UsedH = (unsigned short int) (122);
	KerArg1->InFeatures = (unsigned short int) (4);
	KerArg1->OutFeatures = (unsigned short int) (32);
	KerArg1->TotalInFeatures = (unsigned short int) (16);
	KerArg1->Filter = (signed char * __restrict__) (classification_L1_Memory+4096);
	KerArg1->Out = (int * __restrict__) (classification_L1_Memory+12608);
	KerArg2->In = (int *__restrict__) (classification_L1_Memory+12608);
	KerArg2->Out = (void *__restrict__) (classification_L1_Memory+12608);
	KerArg2->Feat = (unsigned short int) (32);
	KerArg2->W = (unsigned short int) (2);
	KerArg2->H = (unsigned short int) (122);
	KerArg2->Scale = (unsigned char *__restrict__) (classification_L1_Memory+4032);
	KerArg2->ScaleN = (unsigned char *__restrict__) (classification_L1_Memory+4064);
	KerArg2->Infos = (signed char *__restrict__) (classification_L1_Memory+43840);
	KerArg3->In = (signed char * __restrict__) (classification_L1_Memory+12608);
	KerArg3->W = (unsigned short int) (2);
	KerArg3->UsedW = (unsigned short int) (2);
	KerArg3->H = (unsigned short int) (122);
	KerArg3->UsedH = (unsigned short int) (122);
	KerArg3->Feat = (unsigned short int) (32);
	KerArg3->Pad = (v4s) 0;
	KerArg3->PoolMax = (unsigned char) (1);
	KerArg3->DoScale = (unsigned char) (0);
	KerArg3->Infos = (signed char * __restrict__) (classification_L1_Memory+43840);
	/*================================= Read Tiles Prolog ===============================*/
	_C_Out=0; _SC_Out=1952; _LC_Out=1;
	_SP_Out=0;
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+3904), 128, 0, DmaR_Evt1);
	AT_L2_WAIT(0, DmaR_Evt1); /* Wait previous DMA read Bias */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Scale+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+4032), 32, 0, DmaR_Evt2);
	AT_L2_WAIT(0, DmaR_Evt2); /* Wait previous DMA read Scale */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) ScaleN+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+4064), 32, 0, DmaR_Evt3);
	AT_L2_WAIT(0, DmaR_Evt3); /* Wait previous DMA read ScaleN */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Filter+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+4096), 4608, 0, DmaR_Evt4);
	AT_L2_WAIT(0, DmaR_Evt4); /* Wait previous DMA read Filter */
	AT_HYPERRAM_CL_COPY2D(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) In+0), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn+158112+0), 1464, 162, 3, 0, UchanHR1);
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR1); /* Wait previous uDMA read In */
	AT_HYPERRAM_CL_COPY2D(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) In+79056), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn+158112+1952), 1464, 162, 3, 0, UchanHR1);
	AT_L2_COPY(0, ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn+158112+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+0+0), 1464, 0, DmaR_Evt5);
	_NN_In=79056; _SN_In=1464;
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+43840), 13, 0, DmaR_Evt6);
	AT_L2_WAIT(0, DmaR_Evt6); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on D1 */
		int D1Ind_Last = 1;
		for (T0Ind=0; T0Ind<81; T0Ind++, T0Ind_Total++) { /* Iteration on Tile0 */
			int T0Ind_Last = (T0Ind==80), T0Ind_NextLast = ((T0Ind+1)==80), T0Ind_NextNextLast = ((T0Ind+2)==80);
			/*====================== Call Kernel LOC_D0_PROLOG =========================*/
			KerArg0->NormBias = (unsigned char) (((char *)(classification_L1_Memory+43840))[8]);
			AT_FORK(gap_ncore(), (void *) KerParSetBiasB32_SQ8, (void *) KerArg0);
			__CALL(KerParSetBiasB32_SQ8, KerArg0);
			for (D0Ind=0; D0Ind<4; D0Ind++, D0Ind_Total++) { /* Iteration on D0 */
				int D0Ind_Last = (D0Ind==3), D0Ind_NextLast = ((D0Ind+1)==3), D0Ind_NextNextLast = ((D0Ind+2)==3);
				/*================================= Prepare Tiles ===================================*/
				_SNN_In = 0;
				if (!(D0Ind_Last)) {
					if (!(D0Ind_NextLast)) {
						_NN_In = _NN_In + (79056); _LNN_In = ((T0Ind_Last)?3:(4-1*(T0Ind==0))); _SNN_In = (488*_LNN_In); 
					} else if (!(T0Ind_Last)) {
						_NN_In = _NN_In + (2-(1*(T0Ind==0)))+(-237168); _LNN_In = ((T0Ind_NextLast)?3:4); _SNN_In = (488*_LNN_In); 
					} else if (!(1)) {
						_NN_In = _NN_In + (-159)+(-237168); _LNN_In = (3); _SNN_In = (488*_LNN_In); 
					}
				} else if (!(T0Ind_Last)) {
					_NN_In = _NN_In + (79056); _LNN_In = ((T0Ind_NextLast)?3:4); _SNN_In = (488*_LNN_In); 
				} else if (!((1))) {
					_NN_In = _NN_In + (79056); _LNN_In = (3); _SNN_In = (488*_LNN_In); 
				}
				/*============================= End Prepare Tiles ===================================*/
				/*================================= Read Tiles ======================================*/
				AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR1); /* Wait previous uDMA read In */
				if (_SNN_In) {
					AT_HYPERRAM_CL_COPY2D(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) In+_NN_In), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn+158112+1952*((D0Ind_Total)%2)),
							_SNN_In, 162, _LNN_In, 0, UchanHR1);
				}
				AT_L2_WAIT(0, DmaR_Evt5); /* Wait previous DMA read In */
				if (_SN_In) {
					AT_L2_COPY(0, ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn+158112+1952*((D0Ind_Total+1)%2)), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+0+1952*((D0Ind_Total+1)%2)),
							_SN_In, 0, DmaR_Evt5);
				}
				/*============================= End Read Tiles ======================================*/
				/*====================== Call Kernel LOC_D0 =========================*/
				KerArg1->In = (signed char * __restrict__) (classification_L1_Memory+0+1952*((D0Ind_Total)%2));
				KerArg1->W = (unsigned short int) (4-1*(T0Ind==0)-1*(T0Ind_Last));
				KerArg1->UsedW = (unsigned short int) (4-1*(T0Ind==0)-1*(T0Ind_Last));
				KerArg1->Filter = (signed char * __restrict__) (classification_L1_Memory+4096+((D0Ind)*36));
				KerArg1->Pad = (v4s) ((v4s){1*(T0Ind==0),1*(T0Ind_Last),1,1});
				AT_FORK(gap_ncore(), (void *) KerParConv3x3Stride1_SQ8, (void *) KerArg1);
				__CALL(KerParConv3x3Stride1_SQ8, KerArg1);
				/*================================= Update Arg Pipeline =============================*/
				_SN_In = _SNN_In;_LN_In = _LNN_In;
				/*============================= End Update Arg Pipeline =============================*/
			} /* End iteration on D0 */
			/*====================== Call Kernel LOC_D0_EPILOG =========================*/
			AT_FORK(gap_ncore(), (void *) KerParReductIO_CC_SQ8, (void *) KerArg2);
			__CALL(KerParReductIO_CC_SQ8, KerArg2);
			KerArg3->Out = (signed char * __restrict__) (classification_L1_Memory+8704+1952*((T0Ind_Total)%2));
			AT_FORK(gap_ncore(), (void *) KerParPool2x2Stride2_ReLU_SQ8, (void *) KerArg3);
			__CALL(KerParPool2x2Stride2_ReLU_SQ8, KerArg3);
			/*================================= Write Tiles =====================================*/
			if (_SP_Out) AT_L2_WAIT(0, DmaW_Evt1); /* Wait previous DMA write Out */
			AT_L2_COPY2D(0, ((AT_L2_EXT_ADDR_TYPE) Out+_C_Out), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+8704+1952*((T0Ind_Total)%2)),
					_SC_Out, 81, _LC_Out, 1, DmaW_Evt1);
			/*============================= End Write Tiles =====================================*/
			/*================================= Update Arg Pipeline =============================*/
			_SP_Out = _SC_Out;_LP_Out = _LC_Out;
			/*============================= End Update Arg Pipeline =============================*/
			/*================================= Prepare Tiles ===================================*/
			_SC_Out = 0;
			if (!(T0Ind_Last)) {
				_C_Out = _C_Out + (1); _LC_Out = (1); _SC_Out = (1952*_LC_Out); 
			}
			/*============================= End Prepare Tiles ===================================*/
		} /* End iteration on Tile0 */
	} /* End iteration on D1 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_WAIT(0, DmaW_Evt1); /* Wait previous DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S9_Conv2d_64x32x3x3_MaxPool_2x2_Relu(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 44464 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT _DmaW_Evt1, *DmaW_Evt1 = &_DmaW_Evt1;
	AT_L2_EVENT _DmaR_Evt1, *DmaR_Evt1 = &_DmaR_Evt1;
	AT_L2_EVENT _DmaR_Evt2, *DmaR_Evt2 = &_DmaR_Evt2;
	AT_L2_EVENT _DmaR_Evt3, *DmaR_Evt3 = &_DmaR_Evt3;
	AT_L2_EVENT _DmaR_Evt4, *DmaR_Evt4 = &_DmaR_Evt4;
	AT_L2_EVENT _DmaR_Evt5, *DmaR_Evt5 = &_DmaR_Evt5;
	AT_L2_EVENT _DmaR_Evt6, *DmaR_Evt6 = &_DmaR_Evt6;
	AT_HYPERRAM_CL_EVENT _UchanHR1, *UchanHR1 = &_UchanHR1;
	KerSetBias_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;
	KerConv_SQ8_T S_KerArg1, *KerArg1 = &S_KerArg1;
	KerConvLinReduct_SQ8_T S_KerArg2, *KerArg2 = &S_KerArg2;
	KerPool_SQ8_T S_KerArg3, *KerArg3 = &S_KerArg3;

	/* Iteration space related variables */
	int D1Ind, D1Ind_Total=0, D1Ind_Last, D1Ind_NextLast;
	int T0Ind, T0Ind_Total=0, T0Ind_Last, T0Ind_NextLast;
	int D0Ind, D0Ind_Total=0, D0Ind_Last, D0Ind_NextLast;
	/* User kernel arguments related variables */
	unsigned int _P_Out, _C_Out;
	unsigned int _SPP_Out, _SP_Out, _SC_Out;
	unsigned int _LPP_Out, _LP_Out, _LC_Out;
	unsigned int _N_Filter;
	unsigned int _SN_Filter;
	unsigned int _LN_Filter;
	unsigned int _N_In;
	unsigned int _SN_In;
	unsigned int _LN_In;
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D1 Dim: Init: 64, Tiled: 2][Tile0 Dim: 30][D0 Dim: Init: 32, Tiled: 8]
	Ker Arg: Out, Tiled Space: Tile0
		Min Pipe Depth: -2, Max Pipe Depth: 0
		KerArgItSpace: 60 logical tiles, 60 physical tiles
			Total Size: 76800 [D1, [1 x 38400, 38400]][Tile0, 30:[40x1, 28:40x1, 40x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [1 x 38400, 38400]][Tile0, 30:[40x1, 28:40x1, 40x1], 1]
		Tile0: [0, 1280, 40], Tile1: [40, 1280, 40], Tile2; [80, 1280, 40]
	Ker Arg: Bias, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 2 logical tiles, 1 physical tiles
			Total Size: 256 [D1, [1 x 128, 128]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [1 x 128, 128]]
		Tile0: [0, 256, 256], Tile1: [0, 256, 256], Tile2; [0, 256, 256]
	Ker Arg: Scale, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 2 logical tiles, 1 physical tiles
			Total Size: 64 [D1, [1 x 32, 32]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [1 x 32, 32]]
		Tile0: [0, 64, 64], Tile1: [0, 64, 64], Tile2; [0, 64, 64]
	Ker Arg: ScaleN, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 2 logical tiles, 1 physical tiles
			Total Size: 64 [D1, [1 x 32, 32]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [1 x 32, 32]]
		Tile0: [0, 64, 64], Tile1: [0, 64, 64], Tile2; [0, 64, 64]
	Ker Arg: Filter, Tiled Space: D1
		Min Pipe Depth: 0, Max Pipe Depth: 1
		KerArgItSpace: 16 logical tiles, 2 physical tiles
			Total Size: 18432 [D1, [1 x 9216, 9216]][D0, [7 x 1152, 1152]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [1 x 9216, 9216]][D0, [7 x 1152, 1152]]
		Tile0: [0, 9216, 288], Tile1: [9216, 9216, 288], Tile2; [0, 9216, 288]
	Ker Arg: In, Tiled Space: Tile0
		Min Pipe Depth: 0, Max Pipe Depth: 1
		KerArgItSpace: 240 logical tiles, 240 physical tiles
			Total Size: 158112 [D0, [7 x 19764, 19764]][Tile0, 30:[81x3, 28:81x4, 81x4], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 30:[81x3, 6:81x4, 81x4], 1][D0, [7 x 19764, 19764]]
		Tile0: [0, 972, 243], Tile1: [19764, 972, 243], Tile2; [39528, 972, 243]
	Ker Arg: ConvOut, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 60 logical tiles, 1 physical tiles
			Total Size: 1228800 [D1, [1 x 614400, 614400]][Tile0, 30:[80x2, 28:80x2, 80x2], 4]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [1 x 614400, 614400]][Tile0, 30:[80x2, 28:80x2, 80x2], 4]
		Tile0: [0, 20480, 640], Tile1: [0, 20480, 640], Tile2; [0, 20480, 640]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 30 logical tiles, 1 physical tiles
			Total Size: 13 [Tile0, 30:[13x1, 28:13x1, 13x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 30:[13x1, 28:13x1, 13x1], 1]
		Tile0: [0, 13, 13], Tile1: [0, 13, 13], Tile2; [0, 13, 13]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->Out = (int * __restrict__) (classification_L1_Memory+23968);
	KerArg0->W = (unsigned short int) (80);
	KerArg0->H = (unsigned short int) (2);
	KerArg0->Feat = (unsigned short int) (32);
	KerArg1->W = (unsigned short int) (81);
	KerArg1->UsedW = (unsigned short int) (81);
	KerArg1->InFeatures = (unsigned short int) (4);
	KerArg1->OutFeatures = (unsigned short int) (32);
	KerArg1->TotalInFeatures = (unsigned short int) (32);
	KerArg1->Out = (int * __restrict__) (classification_L1_Memory+23968);
	KerArg2->In = (int *__restrict__) (classification_L1_Memory+23968);
	KerArg2->Out = (void *__restrict__) (classification_L1_Memory+23968);
	KerArg2->Feat = (unsigned short int) (32);
	KerArg2->W = (unsigned short int) (80);
	KerArg2->H = (unsigned short int) (2);
	KerArg2->Infos = (signed char *__restrict__) (classification_L1_Memory+44448);
	KerArg3->In = (signed char * __restrict__) (classification_L1_Memory+23968);
	KerArg3->W = (unsigned short int) (80);
	KerArg3->UsedW = (unsigned short int) (80);
	KerArg3->H = (unsigned short int) (2);
	KerArg3->UsedH = (unsigned short int) (2);
	KerArg3->Feat = (unsigned short int) (32);
	KerArg3->Pad = (v4s) 0;
	KerArg3->PoolMax = (unsigned char) (1);
	KerArg3->DoScale = (unsigned char) (0);
	KerArg3->Infos = (signed char * __restrict__) (classification_L1_Memory+44448);
	/*================================= Read Tiles Prolog ===============================*/
	_C_Out=0; _SC_Out=1280; _LC_Out=40;
	_SPP_Out=0; _SP_Out=0;
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+2592), 256, 0, DmaR_Evt1);
	AT_L2_WAIT(0, DmaR_Evt1); /* Wait previous DMA read Bias */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Scale+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+2848), 64, 0, DmaR_Evt2);
	AT_L2_WAIT(0, DmaR_Evt2); /* Wait previous DMA read Scale */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) ScaleN+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+2912), 64, 0, DmaR_Evt3);
	AT_L2_WAIT(0, DmaR_Evt3); /* Wait previous DMA read ScaleN */
	AT_L2_COPY2D(0, ((AT_L2_EXT_ADDR_TYPE) Filter+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+2976+0), 9216, 288, 288, 0, DmaR_Evt4);
	_N_Filter=0;
	AT_L2_COPY2D(0, ((AT_L2_EXT_ADDR_TYPE) In+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+0+0), 972, 4941, 243, 0, DmaR_Evt5);
	_N_In=0;
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+44448), 13, 0, DmaR_Evt6);
	AT_L2_WAIT(0, DmaR_Evt6); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	for (D1Ind=0; D1Ind<2; D1Ind++, D1Ind_Total++) { /* Iteration on D1 */
		int D1Ind_Last = (D1Ind==1), D1Ind_NextLast = ((D1Ind+1)==1);
		/*================================= Prepare Tiles ===================================*/
		_SN_Filter = 0;
		if (!(D1Ind_Last)) {
			_N_Filter = _N_Filter + (9216); _LN_Filter = (288); _SN_Filter = (9216); 
		}
		/*============================= End Prepare Tiles ===================================*/
		/*================================= Read Tiles ======================================*/
		AT_L2_WAIT(0, DmaR_Evt4); /* Wait previous DMA read Filter */
		if (_SN_Filter) {
			AT_L2_COPY2D(0, ((AT_L2_EXT_ADDR_TYPE) Filter+_N_Filter), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+2976+9216*((D1Ind_Total+1)%2)),
					_SN_Filter, 288, _LN_Filter, 0, DmaR_Evt4);
		}
		/*============================= End Read Tiles ======================================*/
		for (T0Ind=0; T0Ind<30; T0Ind++, T0Ind_Total++) { /* Iteration on Tile0 */
			int T0Ind_Last = (T0Ind==29), T0Ind_NextLast = ((T0Ind+1)==29);
			/*====================== Call Kernel LOC_D0_PROLOG =========================*/
			KerArg0->Bias = (void * __restrict__) (classification_L1_Memory+2592+((D1Ind)*128));
			KerArg0->NormBias = (unsigned char) (((char *)(classification_L1_Memory+44448))[8]);
			AT_FORK(gap_ncore(), (void *) KerParSetBiasB32_SQ8, (void *) KerArg0);
			__CALL(KerParSetBiasB32_SQ8, KerArg0);
			for (D0Ind=0; D0Ind<8; D0Ind++, D0Ind_Total++) { /* Iteration on D0 */
				int D0Ind_Last = (D0Ind==7), D0Ind_NextLast = ((D0Ind+1)==7);
				/*================================= Prepare Tiles ===================================*/
				_SN_In = 0;
				if (!(D0Ind_Last)) {
					_N_In = _N_In + (19764); _LN_In = ((T0Ind_Last)?324:(324-81*(T0Ind==0))); _SN_In = (4*_LN_In); 
				} else if (!(T0Ind_Last)) {
					_N_In = _N_In + (162-(81*(T0Ind==0)))+(-138348); _LN_In = ((T0Ind_NextLast)?324:324); _SN_In = (4*_LN_In); 
				} else if (!(D1Ind_Last)) {
					_N_In = _N_In + (-4617)+(-138348); _LN_In = (243); _SN_In = (4*_LN_In); 
				}
				/*============================= End Prepare Tiles ===================================*/
				/*================================= Read Tiles ======================================*/
				AT_L2_WAIT(0, DmaR_Evt5); /* Wait previous DMA read In */
				if (_SN_In) {
					AT_L2_COPY2D(0, ((AT_L2_EXT_ADDR_TYPE) In+_N_In), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+0+1296*((D0Ind_Total+1)%2)),
							_SN_In, 4941, _LN_In, 0, DmaR_Evt5);
				}
				/*============================= End Read Tiles ======================================*/
				/*====================== Call Kernel LOC_D0 =========================*/
				KerArg1->In = (signed char * __restrict__) (classification_L1_Memory+0+1296*((D0Ind_Total)%2));
				KerArg1->H = (unsigned short int) (4-1*(T0Ind==0)-0*(T0Ind_Last));
				KerArg1->UsedH = (unsigned short int) (4-1*(T0Ind==0)-0*(T0Ind_Last));
				KerArg1->Filter = (signed char * __restrict__) (classification_L1_Memory+2976+((D0Ind)*36)+9216*((D1Ind_Total)%2));
				KerArg1->Pad = (v4s) ((v4s){1,0,1*(T0Ind==0),0*(T0Ind_Last)});
				AT_FORK(gap_ncore(), (void *) KerParConv3x3Stride1_SQ8, (void *) KerArg1);
				__CALL(KerParConv3x3Stride1_SQ8, KerArg1);
				/*================================= Update Arg Pipeline =============================*/
				/*============================= End Update Arg Pipeline =============================*/
			} /* End iteration on D0 */
			/*====================== Call Kernel LOC_D0_EPILOG =========================*/
			KerArg2->Scale = (unsigned char *__restrict__) (classification_L1_Memory+2848+((D1Ind)*32));
			KerArg2->ScaleN = (unsigned char *__restrict__) (classification_L1_Memory+2912+((D1Ind)*32));
			AT_FORK(gap_ncore(), (void *) KerParReductIO_CC_SQ8, (void *) KerArg2);
			__CALL(KerParReductIO_CC_SQ8, KerArg2);
			KerArg3->Out = (signed char * __restrict__) (classification_L1_Memory+21408+1280*((T0Ind_Total)%2));
			AT_FORK(gap_ncore(), (void *) KerParPool2x2Stride2_ReLUM_SQ8, (void *) KerArg3);
			__CALL(KerParPool2x2Stride2_ReLUM_SQ8, KerArg3);
			/*================================= Write Tiles =====================================*/
			if (_SP_Out) AT_L2_WAIT(0, DmaW_Evt1); /* Wait previous DMA write Out */
			if (_SPP_Out) AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR1); /* Wait previous uDMA write Out */
			if (_SP_Out) AT_HYPERRAM_CL_COPY2D(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) Out+_P_Out), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn+158112+1280*((T0Ind_Total+-1)%2)),
						_SP_Out, 1200, _LP_Out, 1, UchanHR1);
			AT_L2_COPY(0, ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn+158112+1280*((T0Ind_Total)%2)), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+21408+1280*((T0Ind_Total)%2)),
					_SC_Out, 1, DmaW_Evt1);
			/*============================= End Write Tiles =====================================*/
			/*================================= Update Arg Pipeline =============================*/
			_SPP_Out = _SP_Out;_LPP_Out = _LP_Out;
			_P_Out = _C_Out;_SP_Out = _SC_Out;_LP_Out = _LC_Out;
			/*============================= End Update Arg Pipeline =============================*/
			/*================================= Prepare Tiles ===================================*/
			_SC_Out = 0;
			if (!(T0Ind_Last)) {
				_C_Out = _C_Out + (40); _LC_Out = (40); _SC_Out = (32*_LC_Out); 
			} else if (!(D1Ind_Last)) {
				_C_Out = _C_Out + (38400)+(-1160); _LC_Out = (40); _SC_Out = (32*_LC_Out); 
			}
			/*============================= End Prepare Tiles ===================================*/
		} /* End iteration on Tile0 */
		/*================================= Update Arg Pipeline =============================*/
		/*============================= End Update Arg Pipeline =============================*/
	} /* End iteration on D1 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_WAIT(0, DmaW_Evt1); /* Wait previous DMA write Out */
	if (_SPP_Out) AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR1); /* Wait previous uDMA write Out */
	AT_HYPERRAM_CL_COPY2D(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) Out+_P_Out), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn+158112+1280*((T0Ind_Total+-1)%2)), _SP_Out, 1200, _LP_Out, 1, UchanHR1);
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR1); /* Wait current uDMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S12_Op_FULLY_CONNECTED_0_23_fusion(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 46724 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT _DmaR_Evt1, *DmaR_Evt1 = &_DmaR_Evt1;
	AT_L2_EVENT _DmaR_Evt2, *DmaR_Evt2 = &_DmaR_Evt2;
	AT_L2_EVENT _DmaR_Evt3, *DmaR_Evt3 = &_DmaR_Evt3;
	AT_L2_EVENT _DmaW_Evt1, *DmaW_Evt1 = &_DmaW_Evt1;
	AT_L2_EVENT _DmaR_Evt4, *DmaR_Evt4 = &_DmaR_Evt4;
	AT_L2_EVENT _DmaR_Evt5, *DmaR_Evt5 = &_DmaR_Evt5;
	AT_L2_EVENT _DmaR_Evt6, *DmaR_Evt6 = &_DmaR_Evt6;
	AT_HYPERRAM_CL_EVENT _UchanHR1, *UchanHR1 = &_UchanHR1;
	KerSetBias_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;
	KerLinear_SQ8_T S_KerArg1, *KerArg1 = &S_KerArg1;
	KerConvLinReduct_SQ8_T S_KerArg2, *KerArg2 = &S_KerArg2;

	/* Iteration space related variables */
	int D0Ind, D0Ind_Last, D0Ind_NextLast;
	int T0Ind, T0Ind_Total=0, T0Ind_Last, T0Ind_NextLast, T0Ind_NextNextLast;
	/* User kernel arguments related variables */
	unsigned int _NN_In;
	unsigned int _SN_In, _SNN_In;
	unsigned int _N_Filter;
	unsigned int _SN_Filter;
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D0 Dim: Init: 1, Tiled: 1][Tile0 Dim: 7]
	Ker Arg: In, Tiled Space: Tile0
		Min Pipe Depth: 0, Max Pipe Depth: 2
		KerArgItSpace: 7 logical tiles, 7 physical tiles
			Total Size: 76800 [Tile0, 7:[1x11670, 5:1x11670, 1x6780], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 7:[1x11670, 5:1x11670, 1x6780], 1]
		Tile0: [0, 11670, 11670], Tile1: [11670, 11670, 11670], Tile2; [23340, 11670, 11670]
	Ker Arg: Filter, Tiled Space: Tile0
		Min Pipe Depth: 0, Max Pipe Depth: 1
		KerArgItSpace: 7 logical tiles, 7 physical tiles
			Total Size: 76800 [D0, [0 x 76800, 76800]][Tile0, 7:[1x11670, 5:1x11670, 1x6780], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 76800, 76800]][Tile0, 7:[1x11670, 5:1x11670, 1x6780], 1]
		Tile0: [0, 11670, 11670], Tile1: [11670, 11670, 11670], Tile2; [23340, 11670, 11670]
	Ker Arg: Bias, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4 [D0, [0 x 4, 4]]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 4, 4]]
		Tile0: [0, 4, 4], Tile1: [0, 4, 4], Tile2; [0, 4, 4]
	Ker Arg: LinOut, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4 [D0, [0 x 4, 4]]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 4, 4]]
		Tile0: [0, 4, 4], Tile1: [0, 4, 4], Tile2; [0, 4, 4]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 1 [D0, [0 x 1, 1]]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 1, 1]]
		Tile0: [0, 1, 1], Tile1: [0, 1, 1], Tile2; [0, 1, 1]
	Ker Arg: Scale, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 1 [D0, [0 x 1, 1]]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 1, 1]]
		Tile0: [0, 1, 1], Tile1: [0, 1, 1], Tile2; [0, 1, 1]
	Ker Arg: ScaleN, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 1 [D0, [0 x 1, 1]]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 1, 1]]
		Tile0: [0, 1, 1], Tile1: [0, 1, 1], Tile2; [0, 1, 1]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 7 logical tiles, 1 physical tiles
			Total Size: 13 [Tile0, 7:[1x1, 5:1x1, 1x1], 13]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 7:[1x1, 5:1x1, 1x1], 13]
		Tile0: [0, 13, 13], Tile1: [0, 13, 13], Tile2; [0, 13, 13]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->Out = (int * __restrict__) (classification_L1_Memory+46692);
	KerArg0->W = (unsigned short int) (1);
	KerArg0->H = (unsigned short int) (1);
	KerArg0->Feat = (unsigned short int) (1);
	KerArg0->Bias = (void * __restrict__) (classification_L1_Memory+46688);
	KerArg1->Out = (void * __restrict__) (classification_L1_Memory+46692);
	KerArg1->OutDim = (unsigned short int) (1);
	KerArg1->Infos = (signed char *__restrict__) (classification_L1_Memory+46708);
	KerArg2->In = (int *__restrict__) (classification_L1_Memory+46692);
	KerArg2->Out = (void *__restrict__) (classification_L1_Memory+46696);
	KerArg2->Feat = (unsigned short int) (1);
	KerArg2->W = (unsigned short int) (1);
	KerArg2->H = (unsigned short int) (1);
	KerArg2->Scale = (unsigned char *__restrict__) (classification_L1_Memory+46700);
	KerArg2->ScaleN = (unsigned char *__restrict__) (classification_L1_Memory+46704);
	KerArg2->Infos = (signed char *__restrict__) (classification_L1_Memory+46708);
	/*================================= Read Tiles Prolog ===============================*/
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) In+0), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn+0+0), 11670, 0, UchanHR1);
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR1); /* Wait previous uDMA read In */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) In+11670), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn+0+11672), 11670, 0, UchanHR1);
	AT_L2_COPY(0, ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn+0+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+0+0), 11670, 0, DmaR_Evt1);
	_NN_In=11670; _SN_In=11670;
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Filter+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+23344+0), 11670, 0, DmaR_Evt2);
	_N_Filter=0;
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+46688), 4, 0, DmaR_Evt3);
	AT_L2_WAIT(0, DmaR_Evt3); /* Wait previous DMA read Bias */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Scale+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+46700), 1, 0, DmaR_Evt4);
	AT_L2_WAIT(0, DmaR_Evt4); /* Wait previous DMA read Scale */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) ScaleN+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+46704), 1, 0, DmaR_Evt5);
	AT_L2_WAIT(0, DmaR_Evt5); /* Wait previous DMA read ScaleN */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+46708), 13, 0, DmaR_Evt6);
	AT_L2_WAIT(0, DmaR_Evt6); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on D0 */
		int D0Ind_Last = 1, D0Ind_NextLast = 1;
		/*====================== Call Kernel LOC_LOOP_PROLOG =========================*/
		KerArg0->NormBias = (unsigned char) (((char *)(classification_L1_Memory+46708))[8]);
		AT_FORK(gap_ncore(), (void *) KerParSetBiasB32_SQ8, (void *) KerArg0);
		__CALL(KerParSetBiasB32_SQ8, KerArg0);
		for (T0Ind=0; T0Ind<7; T0Ind++, T0Ind_Total++) { /* Iteration on Tile0 */
			int T0Ind_Last = (T0Ind==6), T0Ind_NextLast = ((T0Ind+1)==6), T0Ind_NextNextLast = ((T0Ind+2)==6);
			/*================================= Prepare Tiles ===================================*/
			_SNN_In = 0;
			if (!(T0Ind_Last)) {
				if (!(T0Ind_NextLast)) {
					_NN_In = _NN_In + (11670); _SNN_In = ((T0Ind_NextNextLast)?6780:11670); 
				} else if (!(1)) {
					_NN_In = _NN_In + (-70020); _SNN_In = (11670); 
				}
			} else if (!((1))) {
				_NN_In = _NN_In + (11670); _SNN_In = (11670); 
			}
			_SN_Filter = 0;
			if (!(T0Ind_Last)) {
				_N_Filter = _N_Filter + (11670); _SN_Filter = (((1)?(((T0Ind_NextLast)?6780:11670)):(((T0Ind_NextLast)?6780:11670)))); 
			}
			/*============================= End Prepare Tiles ===================================*/
			/*================================= Read Tiles ======================================*/
			AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR1); /* Wait previous uDMA read In */
			if (_SNN_In) {
				AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) In+_NN_In), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn+0+11672*((T0Ind_Total)%2)),
						_SNN_In, 0, UchanHR1);
			}
			AT_L2_WAIT(0, DmaR_Evt1); /* Wait previous DMA read In */
			if (_SN_In) {
				AT_L2_COPY(0, ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn+0+11672*((T0Ind_Total+1)%2)), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+0+11672*((T0Ind_Total+1)%2)),
						_SN_In, 0, DmaR_Evt1);
			}
			AT_L2_WAIT(0, DmaR_Evt2); /* Wait previous DMA read Filter */
			if (_SN_Filter) {
				AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Filter+_N_Filter), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+23344+11672*((T0Ind_Total+1)%2)),
						_SN_Filter, 0, DmaR_Evt2);
			}
			/*============================= End Read Tiles ======================================*/
			/*====================== Call Kernel LOC_LOOP =========================*/
			KerArg1->In = (signed char * __restrict__) (classification_L1_Memory+0+11672*((T0Ind_Total)%2));
			KerArg1->Weights = (signed char * __restrict__) (classification_L1_Memory+23344+11672*((T0Ind_Total)%2));
			KerArg1->InDim = (unsigned short int) (T0Ind_Last?6780:11670);
			KerArg1->TotalInDim = (unsigned short int) (T0Ind_Last?6780:11670);
			AT_FORK(gap_ncore(), (void *) KerParLinearLayer_SQ8, (void *) KerArg1);
			__CALL(KerParLinearLayer_SQ8, KerArg1);
			/*================================= Update Arg Pipeline =============================*/
			_SN_In = _SNN_In;
			/*============================= End Update Arg Pipeline =============================*/
		} /* End iteration on Tile0 */
		/*====================== Call Kernel LOC_LOOP_EPILOG =========================*/
		AT_FORK(gap_ncore(), (void *) KerParReduct_CC_ReLU_SQ8, (void *) KerArg2);
		__CALL(KerParReduct_CC_ReLU_SQ8, KerArg2);
	} /* End iteration on D0 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+46696), 1, 1, DmaW_Evt1);
	AT_L2_WAIT(0, DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
int classificationCNN_Construct()

{
	AT_HYPERRAM_CONF_T HyperRamConf;
	AT_HYPERFLASH_FS_CONF_T HyperFlashConf;

	int Error;
	AT_HYPERRAM_CONF_INIT(&HyperRamConf, AT_MEM_L3_HRAM, 0);
	AT_HYPERFLASH_FS_CONF_INIT(&HyperFlashConf, AT_MEM_L3_HFLASH, 0);
	AT_HYPERRAM_OPEN(&HyperRam, &HyperRamConf, &Error);
	if (Error) return 1;
	AT_HYPERFLASH_FS_OPEN(&HyperFlash, &HyperFlashConf, "classification_L3_Flash_Const.dat", &Error);
	if (Error) return 1;

	classification_L3_Memory = (AT_HYPERRAM_POINTER) AT_HYPERRAM_ALLOC(&HyperRam, 339692);
	if (classification_L3_Memory == 0) return 2;
	classification_L2_Memory = (AT_L2_POINTER) AT_L2_ALLOC(0, 82300);
	if (classification_L2_Memory == 0) return 3;
	classification_L2_Memory_Dyn = (AT_L2_POINTER) AT_L2_ALLOC(0, 180448);
	if (classification_L2_Memory_Dyn == 0) return 3;
	classification_L1_Memory = (AT_L1_POINTER) AT_L1_ALLOC(0, 46724);
	if (classification_L1_Memory == 0) return 4;
	AT_HYPERFLASH_FS_FC_EVENT _UchanHF1, *UchanHF1 = &_UchanHF1;
	AT_HYPERRAM_FC_EVENT _UchanHR2, *UchanHR2 = &_UchanHR2;
	/* Moving Dequantize_0_15, size 18432 from HyperFlash at 76800 to (size 18432) HyperRam at 0..18431 */
	{
		int Size = 18432, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 76800 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 0 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving Dequantize_0_3, size 144 from HyperFlash at 100096 to (size 144) L2 at 76800..76943 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 100096), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory + 76800), 144, 0, UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
	/* Moving Sequentialquant_conv2dbiasaddr, size 64 from HyperFlash at 100368 to (size 64) L2 at 77072..77135 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 100368), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory + 77072), 64, 0, UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
	/* Moving S3_Mul_scale, size 16 from HyperFlash at 100624 to (size 16) L2 at 77200..77215 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 100624), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory + 77200), 16, 0, UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
	/* Moving S3_Mul_shift, size 16 from HyperFlash at 100640 to (size 16) L2 at 77216..77231 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 100640), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory + 77216), 16, 0, UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
	/* Moving S3_Infos, size 13 from HyperFlash at 100656 to (size 13) L2 at 77232..77244 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 100656), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory + 77232), 13, 0, UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
	/* Moving Dequantize_0_9, size 4608 from HyperFlash at 95232 to (size 4608) L2 at 77264..81871 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 95232), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory + 77264), 4608, 0, UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
	/* Moving Sequentialquant_conv2d_1biasad, size 128 from HyperFlash at 100240 to (size 128) L2 at 76944..77071 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 100240), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory + 76944), 128, 0, UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
	/* Moving S6_Mul_scale, size 32 from HyperFlash at 100560 to (size 32) L2 at 77136..77167 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 100560), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory + 77136), 32, 0, UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
	/* Moving S6_Mul_shift, size 32 from HyperFlash at 100592 to (size 32) L2 at 77168..77199 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 100592), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory + 77168), 32, 0, UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
	/* Moving S6_Infos, size 13 from HyperFlash at 100672 to (size 13) L2 at 77248..77260 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 100672), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory + 77248), 13, 0, UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
	/* Moving Sequentialquant_conv2d_2biasad, size 256 from HyperFlash at 99840 to (size 256) L2 at 81872..82127 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 99840), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory + 81872), 256, 0, UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
	/* Moving S9_Mul_scale, size 64 from HyperFlash at 100432 to (size 64) L2 at 82128..82191 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 100432), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory + 82128), 64, 0, UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
	/* Moving S9_Mul_shift, size 64 from HyperFlash at 100496 to (size 64) L2 at 82192..82255 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 100496), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory + 82192), 64, 0, UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
	/* Moving S9_Infos, size 13 from HyperFlash at 100688 to (size 16) L2 at 82256..82271 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 100688), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory + 82256), 13, 0, UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
	/* Moving Dequantize_0_22, size 76800 from HyperFlash at 0 to (size 76800) L2 at 0..76799 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 0), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory + 0), 76800, 0, UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
	/* Moving Sequentialquant_densebiasaddre, size 4 from HyperFlash at 100720 to (size 4) L2 at 82288..82291 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 100720), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory + 82288), 4, 0, UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
	/* Moving S12_Mul_scale, size 1 from HyperFlash at 100724 to (size 4) L2 at 82292..82295 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 100724), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory + 82292), 1, 0, UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
	/* Moving S12_Mul_shift, size 1 from HyperFlash at 100728 to (size 4) L2 at 82296..82299 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 100728), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory + 82296), 1, 0, UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
	/* Moving S12_Infos, size 13 from HyperFlash at 100704 to (size 16) L2 at 82272..82287 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 100704), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory + 82272), 13, 0, UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
	return 0;
}
int classificationCNN_Destruct()

{
	AT_HYPERRAM_FREE(&HyperRam, classification_L3_Memory, 339692);
	AT_L2_FREE(0, classification_L2_Memory_Dyn, 180448);
	AT_L2_FREE(0, classification_L2_Memory, 82300);
	AT_L1_FREE(0, classification_L1_Memory, 46724);
	AT_HYPERRAM_CLOSE(&HyperRam);
	AT_HYPERFLASH_FS_CLOSE(&HyperFlash);
	return 0;
}
int classificationCNN_Memory(int Which)

{
	switch (Which) {
		case 0: return 46724;
		case 1: return 180448;
		case 2: return 82300;
	}
	return 0;
}
int classificationCNN(
		signed char * __restrict__ Input_1,
		signed char * __restrict__ Output_1)

{
	AT_HYPERRAM_CL_EVENT _UchanHR0, *UchanHR0 = &_UchanHR0;
	/* Moving Dequantize_0_15, size 18432 from HyperRam at 0 to (size 18432) L2 at 244316 using event 0 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 0), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 162016), 18432, 0, UchanHR0);
	S3_Conv2d_16x1x3x3_MaxPool_2x2_Relu(
		((signed char * __restrict__) Input_1), /* In */
		((signed char * __restrict__) (classification_L2_Memory+76800)), /* Filter */
		((signed int * __restrict__) (classification_L2_Memory+77072)), /* Bias */
		((signed char * __restrict__) (classification_L3_Memory+18432)), /* Out */
		((unsigned char * __restrict__) (classification_L2_Memory+77200)), /* Scale */
		((signed char * __restrict__) (classification_L2_Memory+77216)), /* ScaleN */
		((signed char * __restrict__) (classification_L2_Memory+77232)) /* Infos */
	);
	S6_Conv2d_32x16x3x3_MaxPool_2x2_Relu(
		((signed char * __restrict__) (classification_L3_Memory+18432)), /* In */
		((signed char * __restrict__) (classification_L2_Memory+77264)), /* Filter */
		((signed int * __restrict__) (classification_L2_Memory+76944)), /* Bias */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+0)), /* Out */
		((unsigned char * __restrict__) (classification_L2_Memory+77136)), /* Scale */
		((signed char * __restrict__) (classification_L2_Memory+77168)), /* ScaleN */
		((signed char * __restrict__) (classification_L2_Memory+77248)) /* Infos */
	);
	/* Waiting completion of transfer of Dequantize_0_15 using event 0 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR0);
	S9_Conv2d_64x32x3x3_MaxPool_2x2_Relu(
		((signed char * __restrict__) (classification_L2_Memory_Dyn+0)), /* In */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+162016)), /* Filter */
		((signed int * __restrict__) (classification_L2_Memory+81872)), /* Bias */
		((signed char * __restrict__) (classification_L3_Memory+18432)), /* Out */
		((unsigned char * __restrict__) (classification_L2_Memory+82128)), /* Scale */
		((signed char * __restrict__) (classification_L2_Memory+82192)), /* ScaleN */
		((signed char * __restrict__) (classification_L2_Memory+82256)) /* Infos */
	);
	S12_Op_FULLY_CONNECTED_0_23_fusion(
		((signed char * __restrict__) (classification_L3_Memory+18432)), /* In */
		((signed char * __restrict__) (classification_L2_Memory+0)), /* Filter */
		((signed int * __restrict__) (classification_L2_Memory+82288)), /* Bias */
		((signed char * __restrict__) Output_1), /* Out */
		((unsigned char * __restrict__) (classification_L2_Memory+82292)), /* Scale */
		((signed char * __restrict__) (classification_L2_Memory+82296)), /* ScaleN */
		((signed char * __restrict__) (classification_L2_Memory+82272)) /* Infos */
	);
	return 0;
}
