#include "gatenetKernels.h"
L1_CL_MEM AT_L1_POINTER gatenet_L1_Memory;
L2_MEM AT_L2_POINTER gatenet_L2_Memory;
L2_MEM AT_L2_POINTER gatenet_L2_Memory_Dyn;
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
	/* Shared L1: 27616 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT _DmaW_Evt1, *DmaW_Evt1 = &_DmaW_Evt1;
	AT_L2_EVENT _DmaR_Evt1, *DmaR_Evt1 = &_DmaR_Evt1;
	AT_L2_EVENT _DmaR_Evt2, *DmaR_Evt2 = &_DmaR_Evt2;
	AT_L2_EVENT _DmaR_Evt3, *DmaR_Evt3 = &_DmaR_Evt3;
	AT_L2_EVENT _DmaR_Evt4, *DmaR_Evt4 = &_DmaR_Evt4;
	AT_L2_EVENT _DmaR_Evt5, *DmaR_Evt5 = &_DmaR_Evt5;
	AT_L2_EVENT _DmaR_Evt6, *DmaR_Evt6 = &_DmaR_Evt6;
	KerSetBias_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;
	KerConv_SQ8_T S_KerArg1, *KerArg1 = &S_KerArg1;
	KerConvLinReduct_SQ8_T S_KerArg2, *KerArg2 = &S_KerArg2;
	KerPool_SQ8_T S_KerArg3, *KerArg3 = &S_KerArg3;

	/* Iteration space related variables */
	int D1Ind, D1Ind_Last;
	int T0Ind, T0Ind_Total=0, T0Ind_Last, T0Ind_NextLast;
	int D0Ind, D0Ind_Last;
	/* User kernel arguments related variables */
	unsigned int _C_Out;
	unsigned int _SP_Out, _SC_Out;
	unsigned int _LP_Out, _LC_Out;
	unsigned int _N_In;
	unsigned int _SN_In;
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D1 Dim: Init: 16, Tiled: 1][Tile0 Dim: 60][D0 Dim: Init: 1, Tiled: 1]
	Ker Arg: Out, Tiled Space: Tile0
		Min Pipe Depth: -1, Max Pipe Depth: 0
		KerArgItSpace: 60 logical tiles, 60 physical tiles
			Total Size: 86400 [D1, [0 x 86400, 86400]][Tile0, 60:[90x1, 58:90x1, 90x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 86400, 86400]][Tile0, 60:[90x1, 58:90x1, 90x1], 1]
		Tile0: [0, 1440, 90], Tile1: [90, 1440, 90], Tile2; [180, 1440, 90]
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
		KerArgItSpace: 60 logical tiles, 60 physical tiles
			Total Size: 21600 [D0, [0 x 21600, 21600]][Tile0, 60:[180x3, 58:180x4, 180x3], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 60:[180x2], 1][D0, [0 x 21600, 21600]]
		Tile0: [0, 540, 540], Tile1: [180, 720, 720], Tile2; [540, 720, 720]
	Ker Arg: ConvOut, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 60 logical tiles, 1 physical tiles
			Total Size: 1382400 [D1, [0 x 1382400, 1382400]][Tile0, 60:[180x2, 58:180x2, 180x2], 4]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 1382400, 1382400]][Tile0, 60:[180x2, 58:180x2, 180x2], 4]
		Tile0: [0, 23040, 1440], Tile1: [0, 23040, 1440], Tile2; [0, 23040, 1440]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 60 logical tiles, 1 physical tiles
			Total Size: 13 [Tile0, 60:[13x1, 58:13x1, 13x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 60:[13x1, 58:13x1, 13x1], 1]
		Tile0: [0, 13, 13], Tile1: [0, 13, 13], Tile2; [0, 13, 13]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->Out = (int * __restrict__) (gatenet_L1_Memory+4560);
	KerArg0->W = (unsigned short int) (180);
	KerArg0->H = (unsigned short int) (2);
	KerArg0->Feat = (unsigned short int) (16);
	KerArg0->Bias = (void * __restrict__) (gatenet_L1_Memory+1440);
	KerArg1->W = (unsigned short int) (180);
	KerArg1->UsedW = (unsigned short int) (180);
	KerArg1->InFeatures = (unsigned short int) (1);
	KerArg1->OutFeatures = (unsigned short int) (16);
	KerArg1->TotalInFeatures = (unsigned short int) (1);
	KerArg1->Filter = (signed char * __restrict__) (gatenet_L1_Memory+1536);
	KerArg1->Out = (int * __restrict__) (gatenet_L1_Memory+4560);
	KerArg2->In = (int *__restrict__) (gatenet_L1_Memory+4560);
	KerArg2->Out = (void *__restrict__) (gatenet_L1_Memory+4560);
	KerArg2->Feat = (unsigned short int) (16);
	KerArg2->W = (unsigned short int) (180);
	KerArg2->H = (unsigned short int) (2);
	KerArg2->Scale = (unsigned char *__restrict__) (gatenet_L1_Memory+1504);
	KerArg2->ScaleN = (unsigned char *__restrict__) (gatenet_L1_Memory+1520);
	KerArg2->Infos = (signed char *__restrict__) (gatenet_L1_Memory+27600);
	KerArg3->In = (signed char * __restrict__) (gatenet_L1_Memory+4560);
	KerArg3->W = (unsigned short int) (180);
	KerArg3->UsedW = (unsigned short int) (180);
	KerArg3->H = (unsigned short int) (2);
	KerArg3->UsedH = (unsigned short int) (2);
	KerArg3->Feat = (unsigned short int) (16);
	KerArg3->Pad = (v4s) 0;
	KerArg3->PoolMax = (unsigned char) (1);
	KerArg3->DoScale = (unsigned char) (0);
	KerArg3->Infos = (signed char * __restrict__) (gatenet_L1_Memory+27600);
	/*================================= Read Tiles Prolog ===============================*/
	_C_Out=0; _SC_Out=1440; _LC_Out=90;
	_SP_Out=0;
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) gatenet_L1_Memory+1440), 64, 0, DmaR_Evt1);
	AT_L2_WAIT(0, DmaR_Evt1); /* Wait previous DMA read Bias */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Scale+0), ((AT_L2_INT_ADDR_TYPE) gatenet_L1_Memory+1504), 16, 0, DmaR_Evt2);
	AT_L2_WAIT(0, DmaR_Evt2); /* Wait previous DMA read Scale */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) ScaleN+0), ((AT_L2_INT_ADDR_TYPE) gatenet_L1_Memory+1520), 16, 0, DmaR_Evt3);
	AT_L2_WAIT(0, DmaR_Evt3); /* Wait previous DMA read ScaleN */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Filter+0), ((AT_L2_INT_ADDR_TYPE) gatenet_L1_Memory+1536), 144, 0, DmaR_Evt4);
	AT_L2_WAIT(0, DmaR_Evt4); /* Wait previous DMA read Filter */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In+0), ((AT_L2_INT_ADDR_TYPE) gatenet_L1_Memory+0+0), 540, 0, DmaR_Evt5);
	_N_In=0;
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) gatenet_L1_Memory+27600), 13, 0, DmaR_Evt6);
	AT_L2_WAIT(0, DmaR_Evt6); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on D1 */
		int D1Ind_Last = 1;
		for (T0Ind=0; T0Ind<60; T0Ind++, T0Ind_Total++) { /* Iteration on Tile0 */
			int T0Ind_Last = (T0Ind==59), T0Ind_NextLast = ((T0Ind+1)==59);
			/*================================= Prepare Tiles ===================================*/
			_SN_In = 0;
			if (!(T0Ind_Last)) {
				_N_In = _N_In + (360-(180*(T0Ind==0))); _SN_In = (1*((T0Ind_NextLast)?540:720)); 
			} else if (!(1)) {
				_N_In = _N_In + (-21060); _SN_In = (1*(540)); 
			}
			/*============================= End Prepare Tiles ===================================*/
			/*================================= Read Tiles ======================================*/
			AT_L2_WAIT(0, DmaR_Evt5); /* Wait previous DMA read In */
			if (_SN_In) {
				AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In+_N_In), ((AT_L2_INT_ADDR_TYPE) gatenet_L1_Memory+0+720*((T0Ind_Total+1)%2)),
						_SN_In, 0, DmaR_Evt5);
			}
			/*============================= End Read Tiles ======================================*/
			/*====================== Call Kernel LOC_D0_PROLOG =========================*/
			KerArg0->NormBias = (unsigned char) (((char *)(gatenet_L1_Memory+27600))[8]);
			AT_FORK(gap_ncore(), (void *) KerParSetBiasB32_SQ8, (void *) KerArg0);
			__CALL(KerParSetBiasB32_SQ8, KerArg0);
			{ /* Single iteration on D0 */
				int D0Ind_Last = 1;
				/*====================== Call Kernel LOC_D0 =========================*/
				KerArg1->In = (signed char * __restrict__) (gatenet_L1_Memory+0+720*((T0Ind_Total)%2));
				KerArg1->H = (unsigned short int) (4-1*(T0Ind==0)-1*(T0Ind_Last));
				KerArg1->UsedH = (unsigned short int) (4-1*(T0Ind==0)-1*(T0Ind_Last));
				KerArg1->Pad = (v4s) ((v4s){1,1,1*(T0Ind==0),1*(T0Ind_Last)});
				AT_FORK(gap_ncore(), (void *) KerParConv3x3Stride1_SQ8, (void *) KerArg1);
				__CALL(KerParConv3x3Stride1_SQ8, KerArg1);
			} /* End iteration on D0 */
			/*====================== Call Kernel LOC_D0_EPILOG =========================*/
			AT_FORK(gap_ncore(), (void *) KerParReductIO_CC_SQ8, (void *) KerArg2);
			__CALL(KerParReductIO_CC_SQ8, KerArg2);
			KerArg3->Out = (signed char * __restrict__) (gatenet_L1_Memory+1680+1440*((T0Ind_Total)%2));
			AT_FORK(gap_ncore(), (void *) KerParPool2x2Stride2_ReLU_SQ8, (void *) KerArg3);
			__CALL(KerParPool2x2Stride2_ReLU_SQ8, KerArg3);
			/*================================= Write Tiles =====================================*/
			if (_SP_Out) AT_L2_WAIT(0, DmaW_Evt1); /* Wait previous DMA write Out */
			AT_L2_COPY2D(0, ((AT_L2_EXT_ADDR_TYPE) Out+_C_Out), ((AT_L2_INT_ADDR_TYPE) gatenet_L1_Memory+1680+1440*((T0Ind_Total)%2)),
					_SC_Out, 5400, _LC_Out, 1, DmaW_Evt1);
			/*============================= End Write Tiles =====================================*/
			/*================================= Update Arg Pipeline =============================*/
			_SP_Out = _SC_Out;_LP_Out = _LC_Out;
			/*============================= End Update Arg Pipeline =============================*/
			/*================================= Prepare Tiles ===================================*/
			_SC_Out = 0;
			if (!(T0Ind_Last)) {
				_C_Out = _C_Out + (90); _LC_Out = (90); _SC_Out = (16*_LC_Out); 
			}
			/*============================= End Prepare Tiles ===================================*/
		} /* End iteration on Tile0 */
	} /* End iteration on D1 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_WAIT(0, DmaW_Evt1); /* Wait previous DMA write Out */
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
	/* Shared L1: 45136 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT _DmaW_Evt1, *DmaW_Evt1 = &_DmaW_Evt1;
	AT_L2_EVENT _DmaR_Evt1, *DmaR_Evt1 = &_DmaR_Evt1;
	AT_L2_EVENT _DmaR_Evt2, *DmaR_Evt2 = &_DmaR_Evt2;
	AT_L2_EVENT _DmaR_Evt3, *DmaR_Evt3 = &_DmaR_Evt3;
	AT_L2_EVENT _DmaR_Evt4, *DmaR_Evt4 = &_DmaR_Evt4;
	AT_L2_EVENT _DmaR_Evt5, *DmaR_Evt5 = &_DmaR_Evt5;
	AT_L2_EVENT _DmaR_Evt6, *DmaR_Evt6 = &_DmaR_Evt6;
	KerSetBias_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;
	KerConv_SQ8_T S_KerArg1, *KerArg1 = &S_KerArg1;
	KerConvLinReduct_SQ8_T S_KerArg2, *KerArg2 = &S_KerArg2;
	KerPool_SQ8_T S_KerArg3, *KerArg3 = &S_KerArg3;

	/* Iteration space related variables */
	int D1Ind, D1Ind_Last;
	int T0Ind, T0Ind_Total=0, T0Ind_Last, T0Ind_NextLast;
	int D0Ind, D0Ind_Total=0, D0Ind_Last, D0Ind_NextLast;
	/* User kernel arguments related variables */
	unsigned int _C_Out;
	unsigned int _SP_Out, _SC_Out;
	unsigned int _LP_Out, _LC_Out;
	unsigned int _N_In;
	unsigned int _SN_In;
	unsigned int _LN_In;
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D1 Dim: Init: 32, Tiled: 1][Tile0 Dim: 23][D0 Dim: Init: 16, Tiled: 2]
	Ker Arg: Out, Tiled Space: Tile0
		Min Pipe Depth: -1, Max Pipe Depth: 0
		KerArgItSpace: 23 logical tiles, 23 physical tiles
			Total Size: 43200 [D1, [0 x 43200, 43200]][Tile0, 23:[30x2, 21:30x2, 30x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 43200, 43200]][Tile0, 23:[30x2, 21:30x2, 30x1], 1]
		Tile0: [0, 1920, 2], Tile1: [2, 1920, 2], Tile2; [4, 1920, 2]
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
		KerArgItSpace: 2 logical tiles, 1 physical tiles
			Total Size: 4608 [D1, [0 x 4608, 4608]][D0, [1 x 2304, 2304]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 4608, 4608]][D0, [1 x 2304, 2304]]
		Tile0: [0, 4608, 4608], Tile1: [0, 4608, 4608], Tile2; [0, 4608, 4608]
	Ker Arg: In, Tiled Space: Tile0
		Min Pipe Depth: 0, Max Pipe Depth: 1
		KerArgItSpace: 46 logical tiles, 46 physical tiles
			Total Size: 86400 [D0, [1 x 43200, 43200]][Tile0, 23:[60x5, 21:60x6, 60x3], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 23:[60x5, 60x3], 1][D0, [1 x 43200, 43200]]
		Tile0: [0, 2400, 5], Tile1: [43200, 2400, 5], Tile2; [3, 2880, 6]
	Ker Arg: ConvOut, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 23 logical tiles, 1 physical tiles
			Total Size: 691200 [D1, [0 x 691200, 691200]][Tile0, 23:[60x4, 21:60x4, 60x2], 4]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 691200, 691200]][Tile0, 23:[60x4, 21:60x4, 60x2], 4]
		Tile0: [0, 30720, 16], Tile1: [0, 30720, 16], Tile2; [0, 30720, 16]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 23 logical tiles, 1 physical tiles
			Total Size: 13 [Tile0, 23:[1x13, 21:1x13, 1x13], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 23:[1x13, 21:1x13, 1x13], 1]
		Tile0: [0, 13, 13], Tile1: [0, 13, 13], Tile2; [0, 13, 13]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->Out = (int * __restrict__) (gatenet_L1_Memory+14400);
	KerArg0->H = (unsigned short int) (60);
	KerArg0->Feat = (unsigned short int) (32);
	KerArg0->Bias = (void * __restrict__) (gatenet_L1_Memory+5760);
	KerArg1->H = (unsigned short int) (60);
	KerArg1->UsedH = (unsigned short int) (60);
	KerArg1->InFeatures = (unsigned short int) (8);
	KerArg1->OutFeatures = (unsigned short int) (32);
	KerArg1->TotalInFeatures = (unsigned short int) (16);
	KerArg1->Filter = (signed char * __restrict__) (gatenet_L1_Memory+5952);
	KerArg1->Out = (int * __restrict__) (gatenet_L1_Memory+14400);
	KerArg2->In = (int *__restrict__) (gatenet_L1_Memory+14400);
	KerArg2->Out = (void *__restrict__) (gatenet_L1_Memory+14400);
	KerArg2->Feat = (unsigned short int) (32);
	KerArg2->H = (unsigned short int) (60);
	KerArg2->Scale = (unsigned char *__restrict__) (gatenet_L1_Memory+5888);
	KerArg2->ScaleN = (unsigned char *__restrict__) (gatenet_L1_Memory+5920);
	KerArg2->Infos = (signed char *__restrict__) (gatenet_L1_Memory+45120);
	KerArg3->In = (signed char * __restrict__) (gatenet_L1_Memory+14400);
	KerArg3->H = (unsigned short int) (60);
	KerArg3->UsedH = (unsigned short int) (60);
	KerArg3->Feat = (unsigned short int) (32);
	KerArg3->Pad = (v4s) 0;
	KerArg3->PoolMax = (unsigned char) (1);
	KerArg3->DoScale = (unsigned char) (0);
	KerArg3->Infos = (signed char * __restrict__) (gatenet_L1_Memory+45120);
	/*================================= Read Tiles Prolog ===============================*/
	_C_Out=0; _SC_Out=1920; _LC_Out=2;
	_SP_Out=0;
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) gatenet_L1_Memory+5760), 128, 0, DmaR_Evt1);
	AT_L2_WAIT(0, DmaR_Evt1); /* Wait previous DMA read Bias */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Scale+0), ((AT_L2_INT_ADDR_TYPE) gatenet_L1_Memory+5888), 32, 0, DmaR_Evt2);
	AT_L2_WAIT(0, DmaR_Evt2); /* Wait previous DMA read Scale */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) ScaleN+0), ((AT_L2_INT_ADDR_TYPE) gatenet_L1_Memory+5920), 32, 0, DmaR_Evt3);
	AT_L2_WAIT(0, DmaR_Evt3); /* Wait previous DMA read ScaleN */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Filter+0), ((AT_L2_INT_ADDR_TYPE) gatenet_L1_Memory+5952), 4608, 0, DmaR_Evt4);
	AT_L2_WAIT(0, DmaR_Evt4); /* Wait previous DMA read Filter */
	AT_L2_COPY2D(0, ((AT_L2_EXT_ADDR_TYPE) In+0), ((AT_L2_INT_ADDR_TYPE) gatenet_L1_Memory+0+0), 2400, 90, 5, 0, DmaR_Evt5);
	_N_In=0;
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) gatenet_L1_Memory+45120), 13, 0, DmaR_Evt6);
	AT_L2_WAIT(0, DmaR_Evt6); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on D1 */
		int D1Ind_Last = 1;
		for (T0Ind=0; T0Ind<23; T0Ind++, T0Ind_Total++) { /* Iteration on Tile0 */
			int T0Ind_Last = (T0Ind==22), T0Ind_NextLast = ((T0Ind+1)==22);
			/*====================== Call Kernel LOC_D0_PROLOG =========================*/
			KerArg0->W = (unsigned short int) ((T0Ind_Last)?2:4);
			KerArg0->NormBias = (unsigned char) (((char *)(gatenet_L1_Memory+45120))[8]);
			AT_FORK(gap_ncore(), (void *) KerParSetBiasB32_SQ8, (void *) KerArg0);
			__CALL(KerParSetBiasB32_SQ8, KerArg0);
			for (D0Ind=0; D0Ind<2; D0Ind++, D0Ind_Total++) { /* Iteration on D0 */
				int D0Ind_Last = (D0Ind==1), D0Ind_NextLast = ((D0Ind+1)==1);
				/*================================= Prepare Tiles ===================================*/
				_SN_In = 0;
				if (!(D0Ind_Last)) {
					_N_In = _N_In + (43200); _LN_In = ((T0Ind_Last)?3:(6-1*(T0Ind==0))); _SN_In = (480*_LN_In); 
				} else if (!(T0Ind_Last)) {
					_N_In = _N_In + (4-(1*(T0Ind==0)))+(-43200); _LN_In = ((T0Ind_NextLast)?3:6); _SN_In = (480*_LN_In); 
				} else if (!(1)) {
					_N_In = _N_In + (-87)+(-43200); _LN_In = (5); _SN_In = (480*_LN_In); 
				}
				/*============================= End Prepare Tiles ===================================*/
				/*================================= Read Tiles ======================================*/
				AT_L2_WAIT(0, DmaR_Evt5); /* Wait previous DMA read In */
				if (_SN_In) {
					AT_L2_COPY2D(0, ((AT_L2_EXT_ADDR_TYPE) In+_N_In), ((AT_L2_INT_ADDR_TYPE) gatenet_L1_Memory+0+2880*((D0Ind_Total+1)%2)),
							_SN_In, 90, _LN_In, 0, DmaR_Evt5);
				}
				/*============================= End Read Tiles ======================================*/
				/*====================== Call Kernel LOC_D0 =========================*/
				KerArg1->In = (signed char * __restrict__) (gatenet_L1_Memory+0+2880*((D0Ind_Total)%2));
				KerArg1->W = (unsigned short int) (((T0Ind_Last)?3:6)-1*(T0Ind==0));
				KerArg1->UsedW = (unsigned short int) (((T0Ind_Last)?3:6)-1*(T0Ind==0));
				KerArg1->Filter = (signed char * __restrict__) (gatenet_L1_Memory+5952+((D0Ind)*72));
				KerArg1->Pad = (v4s) ((v4s){1*(T0Ind==0),1*(T0Ind_Last),1,1});
				AT_FORK(gap_ncore(), (void *) KerParConv3x3Stride1_SQ8, (void *) KerArg1);
				__CALL(KerParConv3x3Stride1_SQ8, KerArg1);
				/*================================= Update Arg Pipeline =============================*/
				/*============================= End Update Arg Pipeline =============================*/
			} /* End iteration on D0 */
			/*====================== Call Kernel LOC_D0_EPILOG =========================*/
			KerArg2->W = (unsigned short int) ((T0Ind_Last)?2:4);
			AT_FORK(gap_ncore(), (void *) KerParReductIO_CC_SQ8, (void *) KerArg2);
			__CALL(KerParReductIO_CC_SQ8, KerArg2);
			KerArg3->W = (unsigned short int) ((T0Ind_Last)?2:4);
			KerArg3->UsedW = (unsigned short int) ((T0Ind_Last)?2:4);
			KerArg3->Out = (signed char * __restrict__) (gatenet_L1_Memory+10560+1920*((T0Ind_Total)%2));
			AT_FORK(gap_ncore(), (void *) KerParPool2x2Stride2_ReLU_SQ8, (void *) KerArg3);
			__CALL(KerParPool2x2Stride2_ReLU_SQ8, KerArg3);
			/*================================= Write Tiles =====================================*/
			if (_SP_Out) AT_L2_WAIT(0, DmaW_Evt1); /* Wait previous DMA write Out */
			AT_L2_COPY2D(0, ((AT_L2_EXT_ADDR_TYPE) Out+_C_Out), ((AT_L2_INT_ADDR_TYPE) gatenet_L1_Memory+10560+1920*((T0Ind_Total)%2)),
					_SC_Out, 45, _LC_Out, 1, DmaW_Evt1);
			/*============================= End Write Tiles =====================================*/
			/*================================= Update Arg Pipeline =============================*/
			_SP_Out = _SC_Out;_LP_Out = _LC_Out;
			/*============================= End Update Arg Pipeline =============================*/
			/*================================= Prepare Tiles ===================================*/
			_SC_Out = 0;
			if (!(T0Ind_Last)) {
				_C_Out = _C_Out + (2); _LC_Out = ((T0Ind_NextLast)?1:2); _SC_Out = (960*_LC_Out); 
			}
			/*============================= End Prepare Tiles ===================================*/
		} /* End iteration on Tile0 */
	} /* End iteration on D1 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_WAIT(0, DmaW_Evt1); /* Wait previous DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S9_Conv2d_16x32x3x3_MaxPool_2x2_Relu(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 45040 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT _DmaW_Evt1, *DmaW_Evt1 = &_DmaW_Evt1;
	AT_L2_EVENT _DmaR_Evt1, *DmaR_Evt1 = &_DmaR_Evt1;
	AT_L2_EVENT _DmaR_Evt2, *DmaR_Evt2 = &_DmaR_Evt2;
	AT_L2_EVENT _DmaR_Evt3, *DmaR_Evt3 = &_DmaR_Evt3;
	AT_L2_EVENT _DmaR_Evt4, *DmaR_Evt4 = &_DmaR_Evt4;
	AT_L2_EVENT _DmaR_Evt5, *DmaR_Evt5 = &_DmaR_Evt5;
	AT_L2_EVENT _DmaR_Evt6, *DmaR_Evt6 = &_DmaR_Evt6;
	KerSetBias_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;
	KerConv_SQ8_T S_KerArg1, *KerArg1 = &S_KerArg1;
	KerConvLinReduct_SQ8_T S_KerArg2, *KerArg2 = &S_KerArg2;
	KerPool_SQ8_T S_KerArg3, *KerArg3 = &S_KerArg3;

	/* Iteration space related variables */
	int D1Ind, D1Ind_Last;
	int T0Ind, T0Ind_Total=0, T0Ind_Last, T0Ind_NextLast;
	int D0Ind, D0Ind_Total=0, D0Ind_Last, D0Ind_NextLast;
	/* User kernel arguments related variables */
	unsigned int _C_Out;
	unsigned int _SP_Out, _SC_Out;
	unsigned int _LP_Out, _LC_Out;
	unsigned int _N_In;
	unsigned int _SN_In;
	unsigned int _LN_In;
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D1 Dim: Init: 16, Tiled: 1][Tile0 Dim: 3][D0 Dim: Init: 32, Tiled: 4]
	Ker Arg: Out, Tiled Space: Tile0
		Min Pipe Depth: -1, Max Pipe Depth: 0
		KerArgItSpace: 3 logical tiles, 3 physical tiles
			Total Size: 5280 [D1, [0 x 5280, 5280]][Tile0, 3:[22x5, 1:22x5, 22x5], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 5280, 5280]][Tile0, 3:[22x5, 1:22x5, 22x5], 1]
		Tile0: [0, 1760, 110], Tile1: [110, 1760, 110], Tile2; [220, 1760, 110]
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
		KerArgItSpace: 4 logical tiles, 1 physical tiles
			Total Size: 4608 [D1, [0 x 4608, 4608]][D0, [3 x 1152, 1152]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 4608, 4608]][D0, [3 x 1152, 1152]]
		Tile0: [0, 4608, 4608], Tile1: [0, 4608, 4608], Tile2; [0, 4608, 4608]
	Ker Arg: In, Tiled Space: Tile0
		Min Pipe Depth: 0, Max Pipe Depth: 1
		KerArgItSpace: 12 logical tiles, 12 physical tiles
			Total Size: 43200 [D0, [3 x 10800, 10800]][Tile0, 3:[45x11, 1:45x12, 45x11], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 3:[45x11, 2:45x12, 45x11], 1][D0, [3 x 10800, 10800]]
		Tile0: [0, 3960, 495], Tile1: [10800, 3960, 495], Tile2; [21600, 3960, 495]
	Ker Arg: ConvOut, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 3 logical tiles, 1 physical tiles
			Total Size: 84480 [D1, [0 x 84480, 84480]][Tile0, 3:[44x10, 1:44x10, 44x10], 4]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 84480, 84480]][Tile0, 3:[44x10, 1:44x10, 44x10], 4]
		Tile0: [0, 28160, 1760], Tile1: [0, 28160, 1760], Tile2; [0, 28160, 1760]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 3 logical tiles, 1 physical tiles
			Total Size: 13 [Tile0, 3:[13x1, 1:13x1, 13x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 3:[13x1, 1:13x1, 13x1], 1]
		Tile0: [0, 13, 13], Tile1: [0, 13, 13], Tile2; [0, 13, 13]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->Out = (int * __restrict__) (gatenet_L1_Memory+16864);
	KerArg0->W = (unsigned short int) (44);
	KerArg0->H = (unsigned short int) (10);
	KerArg0->Feat = (unsigned short int) (16);
	KerArg0->Bias = (void * __restrict__) (gatenet_L1_Memory+8640);
	KerArg1->W = (unsigned short int) (45);
	KerArg1->UsedW = (unsigned short int) (45);
	KerArg1->InFeatures = (unsigned short int) (8);
	KerArg1->OutFeatures = (unsigned short int) (16);
	KerArg1->TotalInFeatures = (unsigned short int) (32);
	KerArg1->Filter = (signed char * __restrict__) (gatenet_L1_Memory+8736);
	KerArg1->Out = (int * __restrict__) (gatenet_L1_Memory+16864);
	KerArg2->In = (int *__restrict__) (gatenet_L1_Memory+16864);
	KerArg2->Out = (void *__restrict__) (gatenet_L1_Memory+16864);
	KerArg2->Feat = (unsigned short int) (16);
	KerArg2->W = (unsigned short int) (44);
	KerArg2->H = (unsigned short int) (10);
	KerArg2->Scale = (unsigned char *__restrict__) (gatenet_L1_Memory+8704);
	KerArg2->ScaleN = (unsigned char *__restrict__) (gatenet_L1_Memory+8720);
	KerArg2->Infos = (signed char *__restrict__) (gatenet_L1_Memory+45024);
	KerArg3->In = (signed char * __restrict__) (gatenet_L1_Memory+16864);
	KerArg3->W = (unsigned short int) (44);
	KerArg3->UsedW = (unsigned short int) (44);
	KerArg3->H = (unsigned short int) (10);
	KerArg3->UsedH = (unsigned short int) (10);
	KerArg3->Feat = (unsigned short int) (16);
	KerArg3->Pad = (v4s) 0;
	KerArg3->PoolMax = (unsigned char) (1);
	KerArg3->DoScale = (unsigned char) (0);
	KerArg3->Infos = (signed char * __restrict__) (gatenet_L1_Memory+45024);
	/*================================= Read Tiles Prolog ===============================*/
	_C_Out=0; _SC_Out=1760; _LC_Out=110;
	_SP_Out=0;
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) gatenet_L1_Memory+8640), 64, 0, DmaR_Evt1);
	AT_L2_WAIT(0, DmaR_Evt1); /* Wait previous DMA read Bias */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Scale+0), ((AT_L2_INT_ADDR_TYPE) gatenet_L1_Memory+8704), 16, 0, DmaR_Evt2);
	AT_L2_WAIT(0, DmaR_Evt2); /* Wait previous DMA read Scale */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) ScaleN+0), ((AT_L2_INT_ADDR_TYPE) gatenet_L1_Memory+8720), 16, 0, DmaR_Evt3);
	AT_L2_WAIT(0, DmaR_Evt3); /* Wait previous DMA read ScaleN */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Filter+0), ((AT_L2_INT_ADDR_TYPE) gatenet_L1_Memory+8736), 4608, 0, DmaR_Evt4);
	AT_L2_WAIT(0, DmaR_Evt4); /* Wait previous DMA read Filter */
	AT_L2_COPY2D(0, ((AT_L2_EXT_ADDR_TYPE) In+0), ((AT_L2_INT_ADDR_TYPE) gatenet_L1_Memory+0+0), 3960, 1350, 495, 0, DmaR_Evt5);
	_N_In=0;
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) gatenet_L1_Memory+45024), 13, 0, DmaR_Evt6);
	AT_L2_WAIT(0, DmaR_Evt6); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on D1 */
		int D1Ind_Last = 1;
		for (T0Ind=0; T0Ind<3; T0Ind++, T0Ind_Total++) { /* Iteration on Tile0 */
			int T0Ind_Last = (T0Ind==2), T0Ind_NextLast = ((T0Ind+1)==2);
			/*====================== Call Kernel LOC_D0_PROLOG =========================*/
			KerArg0->NormBias = (unsigned char) (((char *)(gatenet_L1_Memory+45024))[8]);
			AT_FORK(gap_ncore(), (void *) KerParSetBiasB32_SQ8, (void *) KerArg0);
			__CALL(KerParSetBiasB32_SQ8, KerArg0);
			for (D0Ind=0; D0Ind<4; D0Ind++, D0Ind_Total++) { /* Iteration on D0 */
				int D0Ind_Last = (D0Ind==3), D0Ind_NextLast = ((D0Ind+1)==3);
				/*================================= Prepare Tiles ===================================*/
				_SN_In = 0;
				if (!(D0Ind_Last)) {
					_N_In = _N_In + (10800); _LN_In = ((T0Ind_Last)?495:(540-45*(T0Ind==0))); _SN_In = (8*_LN_In); 
				} else if (!(T0Ind_Last)) {
					_N_In = _N_In + (450-(45*(T0Ind==0)))+(-32400); _LN_In = ((T0Ind_NextLast)?495:540); _SN_In = (8*_LN_In); 
				} else if (!(1)) {
					_N_In = _N_In + (-855)+(-32400); _LN_In = (495); _SN_In = (8*_LN_In); 
				}
				/*============================= End Prepare Tiles ===================================*/
				/*================================= Read Tiles ======================================*/
				AT_L2_WAIT(0, DmaR_Evt5); /* Wait previous DMA read In */
				if (_SN_In) {
					AT_L2_COPY2D(0, ((AT_L2_EXT_ADDR_TYPE) In+_N_In), ((AT_L2_INT_ADDR_TYPE) gatenet_L1_Memory+0+4320*((D0Ind_Total+1)%2)),
							_SN_In, 1350, _LN_In, 0, DmaR_Evt5);
				}
				/*============================= End Read Tiles ======================================*/
				/*====================== Call Kernel LOC_D0 =========================*/
				KerArg1->In = (signed char * __restrict__) (gatenet_L1_Memory+0+4320*((D0Ind_Total)%2));
				KerArg1->H = (unsigned short int) (12-1*(T0Ind==0)-1*(T0Ind_Last));
				KerArg1->UsedH = (unsigned short int) (12-1*(T0Ind==0)-1*(T0Ind_Last));
				KerArg1->Filter = (signed char * __restrict__) (gatenet_L1_Memory+8736+((D0Ind)*72));
				KerArg1->Pad = (v4s) ((v4s){1,0,1*(T0Ind==0),1*(T0Ind_Last)});
				AT_FORK(gap_ncore(), (void *) KerParConv3x3Stride1_SQ8, (void *) KerArg1);
				__CALL(KerParConv3x3Stride1_SQ8, KerArg1);
				/*================================= Update Arg Pipeline =============================*/
				/*============================= End Update Arg Pipeline =============================*/
			} /* End iteration on D0 */
			/*====================== Call Kernel LOC_D0_EPILOG =========================*/
			AT_FORK(gap_ncore(), (void *) KerParReductIO_CC_SQ8, (void *) KerArg2);
			__CALL(KerParReductIO_CC_SQ8, KerArg2);
			KerArg3->Out = (signed char * __restrict__) (gatenet_L1_Memory+13344+1760*((T0Ind_Total)%2));
			AT_FORK(gap_ncore(), (void *) KerParPool2x2Stride2_ReLU_SQ8, (void *) KerArg3);
			__CALL(KerParPool2x2Stride2_ReLU_SQ8, KerArg3);
			/*================================= Write Tiles =====================================*/
			if (_SP_Out) AT_L2_WAIT(0, DmaW_Evt1); /* Wait previous DMA write Out */
			AT_L2_COPY2D(0, ((AT_L2_EXT_ADDR_TYPE) Out+_C_Out), ((AT_L2_INT_ADDR_TYPE) gatenet_L1_Memory+13344+1760*((T0Ind_Total)%2)),
					_SC_Out, 330, _LC_Out, 1, DmaW_Evt1);
			/*============================= End Write Tiles =====================================*/
			/*================================= Update Arg Pipeline =============================*/
			_SP_Out = _SC_Out;_LP_Out = _LC_Out;
			/*============================= End Update Arg Pipeline =============================*/
			/*================================= Prepare Tiles ===================================*/
			_SC_Out = 0;
			if (!(T0Ind_Last)) {
				_C_Out = _C_Out + (110); _LC_Out = (110); _SC_Out = (16*_LC_Out); 
			}
			/*============================= End Prepare Tiles ===================================*/
		} /* End iteration on Tile0 */
	} /* End iteration on D1 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_WAIT(0, DmaW_Evt1); /* Wait previous DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S12_Conv2d_16x16x3x3_MaxPool_2x2_Relu(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 28640 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT _DmaR_Evt1, *DmaR_Evt1 = &_DmaR_Evt1;
	AT_L2_EVENT _DmaR_Evt2, *DmaR_Evt2 = &_DmaR_Evt2;
	AT_L2_EVENT _DmaR_Evt3, *DmaR_Evt3 = &_DmaR_Evt3;
	AT_L2_EVENT _DmaR_Evt4, *DmaR_Evt4 = &_DmaR_Evt4;
	AT_L2_EVENT _DmaW_Evt1, *DmaW_Evt1 = &_DmaW_Evt1;
	AT_L2_EVENT _DmaR_Evt5, *DmaR_Evt5 = &_DmaR_Evt5;
	AT_L2_EVENT _DmaR_Evt6, *DmaR_Evt6 = &_DmaR_Evt6;
	KerSetBias_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;
	KerConv_SQ8_T S_KerArg1, *KerArg1 = &S_KerArg1;
	KerConvLinReduct_SQ8_T S_KerArg2, *KerArg2 = &S_KerArg2;
	KerPool_SQ8_T S_KerArg3, *KerArg3 = &S_KerArg3;

	/* Iteration space related variables */
	int D1Ind, D1Ind_Last;
	int T0Ind, T0Ind_Last;
	int D0Ind, D0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D1 Dim: Init: 16, Tiled: 1][Tile0 Dim: 1][D0 Dim: Init: 16, Tiled: 1]
	Ker Arg: ConvOut, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 19712 [D1, [0 x 19712, 19712]][Tile0, 1:[22x14], 4]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 19712, 19712]][Tile0, 1:[22x14], 4]
		Tile0: [0, 19712, 1232], Tile1: [0, 19712, 1232], Tile2; [0, 19712, 1232]
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
			Total Size: 2304 [D1, [0 x 2304, 2304]][D0, [0 x 2304, 2304]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 2304, 2304]][D0, [0 x 2304, 2304]]
		Tile0: [0, 2304, 2304], Tile1: [0, 2304, 2304], Tile2; [0, 2304, 2304]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 1232 [D1, [0 x 1232, 1232]][Tile0, 1:[11x7], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 1232, 1232]][Tile0, 1:[11x7], 1]
		Tile0: [0, 1232, 1232], Tile1: [0, 1232, 1232], Tile2; [0, 1232, 1232]
	Ker Arg: In, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 5280 [D0, [0 x 5280, 5280]][Tile0, 1:[22x15], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[22x15], 1][D0, [0 x 5280, 5280]]
		Tile0: [0, 5280, 5280], Tile1: [0, 5280, 5280], Tile2; [0, 5280, 5280]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 13 [Tile0, 1:[13x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[13x1], 1]
		Tile0: [0, 13, 13], Tile1: [0, 13, 13], Tile2; [0, 13, 13]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->Out = (int * __restrict__) (gatenet_L1_Memory+8912);
	KerArg0->W = (unsigned short int) (22);
	KerArg0->H = (unsigned short int) (14);
	KerArg0->Feat = (unsigned short int) (16);
	KerArg0->Bias = (void * __restrict__) (gatenet_L1_Memory+5280);
	KerArg1->In = (signed char * __restrict__) (gatenet_L1_Memory+0);
	KerArg1->W = (unsigned short int) (22);
	KerArg1->UsedW = (unsigned short int) (22);
	KerArg1->H = (unsigned short int) (15);
	KerArg1->InFeatures = (unsigned short int) (16);
	KerArg1->OutFeatures = (unsigned short int) (16);
	KerArg1->TotalInFeatures = (unsigned short int) (16);
	KerArg1->Filter = (signed char * __restrict__) (gatenet_L1_Memory+5376);
	KerArg1->Out = (int * __restrict__) (gatenet_L1_Memory+8912);
	KerArg1->Pad = (v4s) ((v4s){1,1,1,0});
	KerArg2->In = (int *__restrict__) (gatenet_L1_Memory+8912);
	KerArg2->Out = (void *__restrict__) (gatenet_L1_Memory+8912);
	KerArg2->Feat = (unsigned short int) (16);
	KerArg2->W = (unsigned short int) (22);
	KerArg2->H = (unsigned short int) (14);
	KerArg2->Scale = (unsigned char *__restrict__) (gatenet_L1_Memory+5344);
	KerArg2->ScaleN = (unsigned char *__restrict__) (gatenet_L1_Memory+5360);
	KerArg2->Infos = (signed char *__restrict__) (gatenet_L1_Memory+28624);
	KerArg3->In = (signed char * __restrict__) (gatenet_L1_Memory+8912);
	KerArg3->W = (unsigned short int) (22);
	KerArg3->UsedW = (unsigned short int) (22);
	KerArg3->H = (unsigned short int) (14);
	KerArg3->UsedH = (unsigned short int) (14);
	KerArg3->Feat = (unsigned short int) (16);
	KerArg3->Out = (signed char * __restrict__) (gatenet_L1_Memory+7680);
	KerArg3->Pad = (v4s) 0;
	KerArg3->PoolMax = (unsigned char) (1);
	KerArg3->DoScale = (unsigned char) (0);
	KerArg3->Infos = (signed char * __restrict__) (gatenet_L1_Memory+28624);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) gatenet_L1_Memory+5280), 64, 0, DmaR_Evt1);
	AT_L2_WAIT(0, DmaR_Evt1); /* Wait previous DMA read Bias */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Scale+0), ((AT_L2_INT_ADDR_TYPE) gatenet_L1_Memory+5344), 16, 0, DmaR_Evt2);
	AT_L2_WAIT(0, DmaR_Evt2); /* Wait previous DMA read Scale */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) ScaleN+0), ((AT_L2_INT_ADDR_TYPE) gatenet_L1_Memory+5360), 16, 0, DmaR_Evt3);
	AT_L2_WAIT(0, DmaR_Evt3); /* Wait previous DMA read ScaleN */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Filter+0), ((AT_L2_INT_ADDR_TYPE) gatenet_L1_Memory+5376), 2304, 0, DmaR_Evt4);
	AT_L2_WAIT(0, DmaR_Evt4); /* Wait previous DMA read Filter */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In+0), ((AT_L2_INT_ADDR_TYPE) gatenet_L1_Memory+0), 5280, 0, DmaR_Evt5);
	AT_L2_WAIT(0, DmaR_Evt5); /* Wait previous DMA read In */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) gatenet_L1_Memory+28624), 13, 0, DmaR_Evt6);
	AT_L2_WAIT(0, DmaR_Evt6); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on D1 */
		int D1Ind_Last = 1;
		{ /* Single iteration on Tile0 */
			int T0Ind_Last = 1;
			/*====================== Call Kernel LOC_D0_PROLOG =========================*/
			KerArg0->NormBias = (unsigned char) (((char *)(gatenet_L1_Memory+28624))[8]);
			AT_FORK(gap_ncore(), (void *) KerParSetBiasB32_SQ8, (void *) KerArg0);
			__CALL(KerParSetBiasB32_SQ8, KerArg0);
			{ /* Single iteration on D0 */
				int D0Ind_Last = 1;
				/*====================== Call Kernel LOC_D0 =========================*/
				KerArg1->UsedH = (unsigned short int) (15);
				AT_FORK(gap_ncore(), (void *) KerParConv3x3Stride1_SQ8, (void *) KerArg1);
				__CALL(KerParConv3x3Stride1_SQ8, KerArg1);
			} /* End iteration on D0 */
			/*====================== Call Kernel LOC_D0_EPILOG =========================*/
			AT_FORK(gap_ncore(), (void *) KerParReductIO_CC_SQ8, (void *) KerArg2);
			__CALL(KerParReductIO_CC_SQ8, KerArg2);
			AT_FORK(gap_ncore(), (void *) KerParPool2x2Stride2_ReLU_SQ8, (void *) KerArg3);
			__CALL(KerParPool2x2Stride2_ReLU_SQ8, KerArg3);
		} /* End iteration on Tile0 */
	} /* End iteration on D1 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) gatenet_L1_Memory+7680), 1232, 1, DmaW_Evt1);
	AT_L2_WAIT(0, DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S15_Conv2d_16x16x3x3_MaxPool_2x2_Relu(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 7728 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT _DmaR_Evt1, *DmaR_Evt1 = &_DmaR_Evt1;
	AT_L2_EVENT _DmaR_Evt2, *DmaR_Evt2 = &_DmaR_Evt2;
	AT_L2_EVENT _DmaR_Evt3, *DmaR_Evt3 = &_DmaR_Evt3;
	AT_L2_EVENT _DmaR_Evt4, *DmaR_Evt4 = &_DmaR_Evt4;
	AT_L2_EVENT _DmaW_Evt1, *DmaW_Evt1 = &_DmaW_Evt1;
	AT_L2_EVENT _DmaR_Evt5, *DmaR_Evt5 = &_DmaR_Evt5;
	AT_L2_EVENT _DmaR_Evt6, *DmaR_Evt6 = &_DmaR_Evt6;
	KerSetBias_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;
	KerConv_SQ8_T S_KerArg1, *KerArg1 = &S_KerArg1;
	KerConvLinReduct_SQ8_T S_KerArg2, *KerArg2 = &S_KerArg2;
	KerPool_SQ8_T S_KerArg3, *KerArg3 = &S_KerArg3;

	/* Iteration space related variables */
	int D1Ind, D1Ind_Last;
	int T0Ind, T0Ind_Last;
	int D0Ind, D0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D1 Dim: Init: 16, Tiled: 1][Tile0 Dim: 1][D0 Dim: Init: 16, Tiled: 1]
	Ker Arg: ConvOut, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 3840 [D1, [0 x 3840, 3840]][Tile0, 1:[10x6], 4]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 3840, 3840]][Tile0, 1:[10x6], 4]
		Tile0: [0, 3840, 240], Tile1: [0, 3840, 240], Tile2; [0, 3840, 240]
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
			Total Size: 2304 [D1, [0 x 2304, 2304]][D0, [0 x 2304, 2304]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 2304, 2304]][D0, [0 x 2304, 2304]]
		Tile0: [0, 2304, 2304], Tile1: [0, 2304, 2304], Tile2; [0, 2304, 2304]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 240 [D1, [0 x 240, 240]][Tile0, 1:[5x3], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 240, 240]][Tile0, 1:[5x3], 1]
		Tile0: [0, 240, 240], Tile1: [0, 240, 240], Tile2; [0, 240, 240]
	Ker Arg: In, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 1232 [D0, [0 x 1232, 1232]][Tile0, 1:[11x7], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[11x7], 1][D0, [0 x 1232, 1232]]
		Tile0: [0, 1232, 1232], Tile1: [0, 1232, 1232], Tile2; [0, 1232, 1232]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 13 [Tile0, 1:[13x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[13x1], 1]
		Tile0: [0, 13, 13], Tile1: [0, 13, 13], Tile2; [0, 13, 13]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->Out = (int * __restrict__) (gatenet_L1_Memory+3872);
	KerArg0->W = (unsigned short int) (10);
	KerArg0->H = (unsigned short int) (6);
	KerArg0->Feat = (unsigned short int) (16);
	KerArg0->Bias = (void * __restrict__) (gatenet_L1_Memory+1232);
	KerArg1->In = (signed char * __restrict__) (gatenet_L1_Memory+0);
	KerArg1->W = (unsigned short int) (11);
	KerArg1->UsedW = (unsigned short int) (11);
	KerArg1->H = (unsigned short int) (7);
	KerArg1->InFeatures = (unsigned short int) (16);
	KerArg1->OutFeatures = (unsigned short int) (16);
	KerArg1->TotalInFeatures = (unsigned short int) (16);
	KerArg1->Filter = (signed char * __restrict__) (gatenet_L1_Memory+1328);
	KerArg1->Out = (int * __restrict__) (gatenet_L1_Memory+3872);
	KerArg1->Pad = (v4s) ((v4s){1,0,1,0});
	KerArg2->In = (int *__restrict__) (gatenet_L1_Memory+3872);
	KerArg2->Out = (void *__restrict__) (gatenet_L1_Memory+3872);
	KerArg2->Feat = (unsigned short int) (16);
	KerArg2->W = (unsigned short int) (10);
	KerArg2->H = (unsigned short int) (6);
	KerArg2->Scale = (unsigned char *__restrict__) (gatenet_L1_Memory+1296);
	KerArg2->ScaleN = (unsigned char *__restrict__) (gatenet_L1_Memory+1312);
	KerArg2->Infos = (signed char *__restrict__) (gatenet_L1_Memory+7712);
	KerArg3->In = (signed char * __restrict__) (gatenet_L1_Memory+3872);
	KerArg3->W = (unsigned short int) (10);
	KerArg3->UsedW = (unsigned short int) (10);
	KerArg3->H = (unsigned short int) (6);
	KerArg3->UsedH = (unsigned short int) (6);
	KerArg3->Feat = (unsigned short int) (16);
	KerArg3->Out = (signed char * __restrict__) (gatenet_L1_Memory+3632);
	KerArg3->Pad = (v4s) 0;
	KerArg3->PoolMax = (unsigned char) (1);
	KerArg3->DoScale = (unsigned char) (0);
	KerArg3->Infos = (signed char * __restrict__) (gatenet_L1_Memory+7712);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) gatenet_L1_Memory+1232), 64, 0, DmaR_Evt1);
	AT_L2_WAIT(0, DmaR_Evt1); /* Wait previous DMA read Bias */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Scale+0), ((AT_L2_INT_ADDR_TYPE) gatenet_L1_Memory+1296), 16, 0, DmaR_Evt2);
	AT_L2_WAIT(0, DmaR_Evt2); /* Wait previous DMA read Scale */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) ScaleN+0), ((AT_L2_INT_ADDR_TYPE) gatenet_L1_Memory+1312), 16, 0, DmaR_Evt3);
	AT_L2_WAIT(0, DmaR_Evt3); /* Wait previous DMA read ScaleN */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Filter+0), ((AT_L2_INT_ADDR_TYPE) gatenet_L1_Memory+1328), 2304, 0, DmaR_Evt4);
	AT_L2_WAIT(0, DmaR_Evt4); /* Wait previous DMA read Filter */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In+0), ((AT_L2_INT_ADDR_TYPE) gatenet_L1_Memory+0), 1232, 0, DmaR_Evt5);
	AT_L2_WAIT(0, DmaR_Evt5); /* Wait previous DMA read In */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) gatenet_L1_Memory+7712), 13, 0, DmaR_Evt6);
	AT_L2_WAIT(0, DmaR_Evt6); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on D1 */
		int D1Ind_Last = 1;
		{ /* Single iteration on Tile0 */
			int T0Ind_Last = 1;
			/*====================== Call Kernel LOC_D0_PROLOG =========================*/
			KerArg0->NormBias = (unsigned char) (((char *)(gatenet_L1_Memory+7712))[8]);
			AT_FORK(gap_ncore(), (void *) KerParSetBiasB32_SQ8, (void *) KerArg0);
			__CALL(KerParSetBiasB32_SQ8, KerArg0);
			{ /* Single iteration on D0 */
				int D0Ind_Last = 1;
				/*====================== Call Kernel LOC_D0 =========================*/
				KerArg1->UsedH = (unsigned short int) (7);
				AT_FORK(gap_ncore(), (void *) KerParConv3x3Stride1_SQ8, (void *) KerArg1);
				__CALL(KerParConv3x3Stride1_SQ8, KerArg1);
			} /* End iteration on D0 */
			/*====================== Call Kernel LOC_D0_EPILOG =========================*/
			AT_FORK(gap_ncore(), (void *) KerParReductIO_CC_SQ8, (void *) KerArg2);
			__CALL(KerParReductIO_CC_SQ8, KerArg2);
			AT_FORK(gap_ncore(), (void *) KerParPool2x2Stride2_ReLU_SQ8, (void *) KerArg3);
			__CALL(KerParPool2x2Stride2_ReLU_SQ8, KerArg3);
		} /* End iteration on Tile0 */
	} /* End iteration on D1 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) gatenet_L1_Memory+3632), 240, 1, DmaW_Evt1);
	AT_L2_WAIT(0, DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S18_Conv2d_16x16x3x3_Relu(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 3856 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT _DmaR_Evt1, *DmaR_Evt1 = &_DmaR_Evt1;
	AT_L2_EVENT _DmaR_Evt2, *DmaR_Evt2 = &_DmaR_Evt2;
	AT_L2_EVENT _DmaR_Evt3, *DmaR_Evt3 = &_DmaR_Evt3;
	AT_L2_EVENT _DmaR_Evt4, *DmaR_Evt4 = &_DmaR_Evt4;
	AT_L2_EVENT _DmaW_Evt1, *DmaW_Evt1 = &_DmaW_Evt1;
	AT_L2_EVENT _DmaR_Evt5, *DmaR_Evt5 = &_DmaR_Evt5;
	AT_L2_EVENT _DmaR_Evt6, *DmaR_Evt6 = &_DmaR_Evt6;
	KerSetBias_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;
	KerConv_SQ8_T S_KerArg1, *KerArg1 = &S_KerArg1;
	KerConvLinReduct_SQ8_T S_KerArg2, *KerArg2 = &S_KerArg2;

	/* Iteration space related variables */
	int D1Ind, D1Ind_Last;
	int T0Ind, T0Ind_Last;
	int D0Ind, D0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D1 Dim: Init: 16, Tiled: 1][Tile0 Dim: 1][D0 Dim: Init: 16, Tiled: 1]
	Ker Arg: ConvOut, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 960 [D1, [0 x 960, 960]][Tile0, 1:[5x3], 4]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 960, 960]][Tile0, 1:[5x3], 4]
		Tile0: [0, 960, 60], Tile1: [0, 960, 60], Tile2; [0, 960, 60]
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
			Total Size: 2304 [D1, [0 x 2304, 2304]][D0, [0 x 2304, 2304]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 2304, 2304]][D0, [0 x 2304, 2304]]
		Tile0: [0, 2304, 2304], Tile1: [0, 2304, 2304], Tile2; [0, 2304, 2304]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 240 [D1, [0 x 240, 240]][Tile0, 1:[5x3], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 240, 240]][Tile0, 1:[5x3], 1]
		Tile0: [0, 240, 240], Tile1: [0, 240, 240], Tile2; [0, 240, 240]
	Ker Arg: In, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 240 [D0, [0 x 240, 240]][Tile0, 1:[5x3], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[5x3], 1][D0, [0 x 240, 240]]
		Tile0: [0, 240, 240], Tile1: [0, 240, 240], Tile2; [0, 240, 240]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 13 [Tile0, 1:[13x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[13x1], 1]
		Tile0: [0, 13, 13], Tile1: [0, 13, 13], Tile2; [0, 13, 13]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->Out = (int * __restrict__) (gatenet_L1_Memory+2880);
	KerArg0->W = (unsigned short int) (5);
	KerArg0->H = (unsigned short int) (3);
	KerArg0->Feat = (unsigned short int) (16);
	KerArg0->Bias = (void * __restrict__) (gatenet_L1_Memory+240);
	KerArg1->In = (signed char * __restrict__) (gatenet_L1_Memory+0);
	KerArg1->W = (unsigned short int) (5);
	KerArg1->UsedW = (unsigned short int) (5);
	KerArg1->H = (unsigned short int) (3);
	KerArg1->InFeatures = (unsigned short int) (16);
	KerArg1->OutFeatures = (unsigned short int) (16);
	KerArg1->TotalInFeatures = (unsigned short int) (16);
	KerArg1->Filter = (signed char * __restrict__) (gatenet_L1_Memory+336);
	KerArg1->Out = (int * __restrict__) (gatenet_L1_Memory+2880);
	KerArg1->Pad = (v4s) ((v4s){1,1,1,1});
	KerArg2->In = (int *__restrict__) (gatenet_L1_Memory+2880);
	KerArg2->Out = (void *__restrict__) (gatenet_L1_Memory+2640);
	KerArg2->Feat = (unsigned short int) (16);
	KerArg2->W = (unsigned short int) (5);
	KerArg2->H = (unsigned short int) (3);
	KerArg2->Scale = (unsigned char *__restrict__) (gatenet_L1_Memory+304);
	KerArg2->ScaleN = (unsigned char *__restrict__) (gatenet_L1_Memory+320);
	KerArg2->Infos = (signed char *__restrict__) (gatenet_L1_Memory+3840);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) gatenet_L1_Memory+240), 64, 0, DmaR_Evt1);
	AT_L2_WAIT(0, DmaR_Evt1); /* Wait previous DMA read Bias */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Scale+0), ((AT_L2_INT_ADDR_TYPE) gatenet_L1_Memory+304), 16, 0, DmaR_Evt2);
	AT_L2_WAIT(0, DmaR_Evt2); /* Wait previous DMA read Scale */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) ScaleN+0), ((AT_L2_INT_ADDR_TYPE) gatenet_L1_Memory+320), 16, 0, DmaR_Evt3);
	AT_L2_WAIT(0, DmaR_Evt3); /* Wait previous DMA read ScaleN */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Filter+0), ((AT_L2_INT_ADDR_TYPE) gatenet_L1_Memory+336), 2304, 0, DmaR_Evt4);
	AT_L2_WAIT(0, DmaR_Evt4); /* Wait previous DMA read Filter */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In+0), ((AT_L2_INT_ADDR_TYPE) gatenet_L1_Memory+0), 240, 0, DmaR_Evt5);
	AT_L2_WAIT(0, DmaR_Evt5); /* Wait previous DMA read In */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) gatenet_L1_Memory+3840), 13, 0, DmaR_Evt6);
	AT_L2_WAIT(0, DmaR_Evt6); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on D1 */
		int D1Ind_Last = 1;
		{ /* Single iteration on Tile0 */
			int T0Ind_Last = 1;
			/*====================== Call Kernel LOC_D0_PROLOG =========================*/
			KerArg0->NormBias = (unsigned char) (((char *)(gatenet_L1_Memory+3840))[8]);
			AT_FORK(gap_ncore(), (void *) KerParSetBiasB32_SQ8, (void *) KerArg0);
			__CALL(KerParSetBiasB32_SQ8, KerArg0);
			{ /* Single iteration on D0 */
				int D0Ind_Last = 1;
				/*====================== Call Kernel LOC_D0 =========================*/
				KerArg1->UsedH = (unsigned short int) (3);
				AT_FORK(gap_ncore(), (void *) KerParConv3x3Stride1_SQ8, (void *) KerArg1);
				__CALL(KerParConv3x3Stride1_SQ8, KerArg1);
			} /* End iteration on D0 */
			/*====================== Call Kernel LOC_D0_EPILOG =========================*/
			AT_FORK(gap_ncore(), (void *) KerParReduct_CC_ReLUM_SQ8, (void *) KerArg2);
			__CALL(KerParReduct_CC_ReLUM_SQ8, KerArg2);
		} /* End iteration on Tile0 */
	} /* End iteration on D1 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) gatenet_L1_Memory+2640), 240, 1, DmaW_Evt1);
	AT_L2_WAIT(0, DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S21_Op_FULLY_CONNECTED_0_13_fusion(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 1000 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT _DmaR_Evt1, *DmaR_Evt1 = &_DmaR_Evt1;
	AT_L2_EVENT _DmaR_Evt2, *DmaR_Evt2 = &_DmaR_Evt2;
	AT_L2_EVENT _DmaR_Evt3, *DmaR_Evt3 = &_DmaR_Evt3;
	AT_L2_EVENT _DmaW_Evt1, *DmaW_Evt1 = &_DmaW_Evt1;
	AT_L2_EVENT _DmaR_Evt4, *DmaR_Evt4 = &_DmaR_Evt4;
	AT_L2_EVENT _DmaR_Evt5, *DmaR_Evt5 = &_DmaR_Evt5;
	AT_L2_EVENT _DmaR_Evt6, *DmaR_Evt6 = &_DmaR_Evt6;
	KerLinear_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Iteration space related variables */
	int D0Ind, D0Ind_Last;
	int T0Ind, T0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D0 Dim: Init: 3, Tiled: 1][Tile0 Dim: 1]
	Ker Arg: In, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 240 [Tile0, 1:[1x1], 240]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x1], 240]
		Tile0: [0, 240, 240], Tile1: [0, 240, 240], Tile2; [0, 240, 240]
	Ker Arg: Filter, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 720 [D0, [0 x 720, 720]]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 720, 720]]
		Tile0: [0, 720, 720], Tile1: [0, 720, 720], Tile2; [0, 720, 720]
	Ker Arg: Bias, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 12 [D0, [0 x 12, 12]]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 12, 12]]
		Tile0: [0, 12, 12], Tile1: [0, 12, 12], Tile2; [0, 12, 12]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 3 [D0, [0 x 3, 3]]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 3, 3]]
		Tile0: [0, 3, 3], Tile1: [0, 3, 3], Tile2; [0, 3, 3]
	Ker Arg: Scale, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 3 [D0, [0 x 3, 3]]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 3, 3]]
		Tile0: [0, 3, 3], Tile1: [0, 3, 3], Tile2; [0, 3, 3]
	Ker Arg: ScaleN, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 3 [D0, [0 x 3, 3]]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 3, 3]]
		Tile0: [0, 3, 3], Tile1: [0, 3, 3], Tile2; [0, 3, 3]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 13 [Tile0, 1:[1x1], 13]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x1], 13]
		Tile0: [0, 13, 13], Tile1: [0, 13, 13], Tile2; [0, 13, 13]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->In = (signed char * __restrict__) (gatenet_L1_Memory+0);
	KerArg0->Weights = (signed char * __restrict__) (gatenet_L1_Memory+240);
	KerArg0->Bias = (void * __restrict__) (gatenet_L1_Memory+960);
	KerArg0->Out = (void * __restrict__) (gatenet_L1_Memory+972);
	KerArg0->InDim = (unsigned short int) (240);
	KerArg0->TotalInDim = (unsigned short int) (240);
	KerArg0->OutDim = (unsigned short int) (3);
	KerArg0->Scale = (unsigned char *__restrict__) (gatenet_L1_Memory+976);
	KerArg0->ScaleN = (unsigned char *__restrict__) (gatenet_L1_Memory+980);
	KerArg0->Infos = (signed char *__restrict__) (gatenet_L1_Memory+984);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In+0), ((AT_L2_INT_ADDR_TYPE) gatenet_L1_Memory+0), 240, 0, DmaR_Evt1);
	AT_L2_WAIT(0, DmaR_Evt1); /* Wait previous DMA read In */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Filter+0), ((AT_L2_INT_ADDR_TYPE) gatenet_L1_Memory+240), 720, 0, DmaR_Evt2);
	AT_L2_WAIT(0, DmaR_Evt2); /* Wait previous DMA read Filter */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) gatenet_L1_Memory+960), 12, 0, DmaR_Evt3);
	AT_L2_WAIT(0, DmaR_Evt3); /* Wait previous DMA read Bias */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Scale+0), ((AT_L2_INT_ADDR_TYPE) gatenet_L1_Memory+976), 3, 0, DmaR_Evt4);
	AT_L2_WAIT(0, DmaR_Evt4); /* Wait previous DMA read Scale */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) ScaleN+0), ((AT_L2_INT_ADDR_TYPE) gatenet_L1_Memory+980), 3, 0, DmaR_Evt5);
	AT_L2_WAIT(0, DmaR_Evt5); /* Wait previous DMA read ScaleN */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) gatenet_L1_Memory+984), 13, 0, DmaR_Evt6);
	AT_L2_WAIT(0, DmaR_Evt6); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on D0 */
		int D0Ind_Last = 1;
		{ /* Single iteration on Tile0 */
			int T0Ind_Last = 1;
			/*====================== Call Kernel LOC_LOOP =========================*/
			AT_FORK(gap_ncore(), (void *) KerParLinearLayerFullFeatB32_ReLU_SQ8, (void *) KerArg0);
			__CALL(KerParLinearLayerFullFeatB32_ReLU_SQ8, KerArg0);
		} /* End iteration on Tile0 */
	} /* End iteration on D0 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) gatenet_L1_Memory+972), 3, 1, DmaW_Evt1);
	AT_L2_WAIT(0, DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
int gatenetCNN_Construct()

{
	AT_HYPERFLASH_FS_CONF_T HyperFlashConf;

	int Error;
	AT_HYPERFLASH_FS_CONF_INIT(&HyperFlashConf, AT_MEM_L3_HFLASH, 0);
	AT_HYPERFLASH_FS_OPEN(&HyperFlash, &HyperFlashConf, "gatenet_L3_Flash_Const.dat", &Error);
	if (Error) return 1;

	gatenet_L2_Memory = (AT_L2_POINTER) AT_L2_ALLOC(0, 17796);
	if (gatenet_L2_Memory == 0) return 3;
	gatenet_L2_Memory_Dyn = (AT_L2_POINTER) AT_L2_ALLOC(0, 129600);
	if (gatenet_L2_Memory_Dyn == 0) return 3;
	gatenet_L1_Memory = (AT_L1_POINTER) AT_L1_ALLOC(0, 45136);
	if (gatenet_L1_Memory == 0) return 4;
	AT_HYPERFLASH_FS_FC_EVENT _UchanHF1, *UchanHF1 = &_UchanHF1;
	/* Moving Modelquant_conv1conv2dmodelqua, size 144 from HyperFlash at 16848 to (size 144) L2 at 16848..16991 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) gatenet_L3_Flash + 16848), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) gatenet_L2_Memory + 16848), 144, 0, UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
	/* Moving Modelquant_activationrelumodel, size 64 from HyperFlash at 17120 to (size 64) L2 at 17120..17183 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) gatenet_L3_Flash + 17120), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) gatenet_L2_Memory + 17120), 64, 0, UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
	/* Moving S3_Mul_scale, size 16 from HyperFlash at 17504 to (size 16) L2 at 17504..17519 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) gatenet_L3_Flash + 17504), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) gatenet_L2_Memory + 17504), 16, 0, UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
	/* Moving S3_Mul_shift, size 16 from HyperFlash at 17520 to (size 16) L2 at 17520..17535 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) gatenet_L3_Flash + 17520), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) gatenet_L2_Memory + 17520), 16, 0, UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
	/* Moving S3_Infos, size 13 from HyperFlash at 17536 to (size 13) L2 at 17536..17548 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) gatenet_L3_Flash + 17536), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) gatenet_L2_Memory + 17536), 13, 0, UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
	/* Moving Modelquant_conv2conv2dmodelqua, size 4608 from HyperFlash at 0 to (size 4608) L2 at 0..4607 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) gatenet_L3_Flash + 0), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) gatenet_L2_Memory + 0), 4608, 0, UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
	/* Moving Modelquant_activation_1relumod, size 128 from HyperFlash at 16992 to (size 128) L2 at 16992..17119 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) gatenet_L3_Flash + 16992), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) gatenet_L2_Memory + 16992), 128, 0, UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
	/* Moving S6_Mul_scale, size 32 from HyperFlash at 17440 to (size 32) L2 at 17440..17471 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) gatenet_L3_Flash + 17440), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) gatenet_L2_Memory + 17440), 32, 0, UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
	/* Moving S6_Mul_shift, size 32 from HyperFlash at 17472 to (size 32) L2 at 17472..17503 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) gatenet_L3_Flash + 17472), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) gatenet_L2_Memory + 17472), 32, 0, UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
	/* Moving S6_Infos, size 13 from HyperFlash at 17552 to (size 13) L2 at 17552..17564 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) gatenet_L3_Flash + 17552), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) gatenet_L2_Memory + 17552), 13, 0, UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
	/* Moving Modelquant_conv3conv2dmodelqua, size 4608 from HyperFlash at 4608 to (size 4608) L2 at 4608..9215 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) gatenet_L3_Flash + 4608), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) gatenet_L2_Memory + 4608), 4608, 0, UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
	/* Moving Modelquant_activation_2relumod, size 64 from HyperFlash at 17184 to (size 64) L2 at 17184..17247 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) gatenet_L3_Flash + 17184), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) gatenet_L2_Memory + 17184), 64, 0, UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
	/* Moving S9_Mul_scale, size 16 from HyperFlash at 17568 to (size 16) L2 at 17568..17583 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) gatenet_L3_Flash + 17568), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) gatenet_L2_Memory + 17568), 16, 0, UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
	/* Moving S9_Mul_shift, size 16 from HyperFlash at 17584 to (size 16) L2 at 17584..17599 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) gatenet_L3_Flash + 17584), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) gatenet_L2_Memory + 17584), 16, 0, UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
	/* Moving S9_Infos, size 13 from HyperFlash at 17600 to (size 13) L2 at 17600..17612 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) gatenet_L3_Flash + 17600), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) gatenet_L2_Memory + 17600), 13, 0, UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
	/* Moving Modelquant_conv4conv2dmodelqua, size 2304 from HyperFlash at 9216 to (size 2304) L2 at 9216..11519 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) gatenet_L3_Flash + 9216), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) gatenet_L2_Memory + 9216), 2304, 0, UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
	/* Moving Modelquant_activation_3relumod, size 64 from HyperFlash at 17248 to (size 64) L2 at 17248..17311 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) gatenet_L3_Flash + 17248), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) gatenet_L2_Memory + 17248), 64, 0, UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
	/* Moving S12_Mul_scale, size 16 from HyperFlash at 17616 to (size 16) L2 at 17616..17631 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) gatenet_L3_Flash + 17616), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) gatenet_L2_Memory + 17616), 16, 0, UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
	/* Moving S12_Mul_shift, size 16 from HyperFlash at 17632 to (size 16) L2 at 17632..17647 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) gatenet_L3_Flash + 17632), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) gatenet_L2_Memory + 17632), 16, 0, UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
	/* Moving S12_Infos, size 13 from HyperFlash at 17648 to (size 13) L2 at 17648..17660 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) gatenet_L3_Flash + 17648), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) gatenet_L2_Memory + 17648), 13, 0, UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
	/* Moving Modelquant_conv5conv2dmodelqua, size 2304 from HyperFlash at 11520 to (size 2304) L2 at 11520..13823 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) gatenet_L3_Flash + 11520), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) gatenet_L2_Memory + 11520), 2304, 0, UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
	/* Moving Modelquant_activation_4relumod, size 64 from HyperFlash at 17312 to (size 64) L2 at 17312..17375 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) gatenet_L3_Flash + 17312), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) gatenet_L2_Memory + 17312), 64, 0, UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
	/* Moving S15_Mul_scale, size 16 from HyperFlash at 17664 to (size 16) L2 at 17664..17679 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) gatenet_L3_Flash + 17664), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) gatenet_L2_Memory + 17664), 16, 0, UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
	/* Moving S15_Mul_shift, size 16 from HyperFlash at 17680 to (size 16) L2 at 17680..17695 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) gatenet_L3_Flash + 17680), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) gatenet_L2_Memory + 17680), 16, 0, UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
	/* Moving S15_Infos, size 13 from HyperFlash at 17696 to (size 13) L2 at 17696..17708 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) gatenet_L3_Flash + 17696), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) gatenet_L2_Memory + 17696), 13, 0, UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
	/* Moving Modelquant_conv6conv2dmodelqua, size 2304 from HyperFlash at 13824 to (size 2304) L2 at 13824..16127 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) gatenet_L3_Flash + 13824), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) gatenet_L2_Memory + 13824), 2304, 0, UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
	/* Moving Modelquant_activation_5relumod, size 64 from HyperFlash at 17376 to (size 64) L2 at 17376..17439 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) gatenet_L3_Flash + 17376), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) gatenet_L2_Memory + 17376), 64, 0, UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
	/* Moving S18_Mul_scale, size 16 from HyperFlash at 17712 to (size 16) L2 at 17712..17727 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) gatenet_L3_Flash + 17712), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) gatenet_L2_Memory + 17712), 16, 0, UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
	/* Moving S18_Mul_shift, size 16 from HyperFlash at 17728 to (size 16) L2 at 17728..17743 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) gatenet_L3_Flash + 17728), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) gatenet_L2_Memory + 17728), 16, 0, UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
	/* Moving S18_Infos, size 13 from HyperFlash at 17744 to (size 13) L2 at 17744..17756 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) gatenet_L3_Flash + 17744), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) gatenet_L2_Memory + 17744), 13, 0, UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
	/* Moving Modelquant_densematmulmodelqua, size 720 from HyperFlash at 16128 to (size 720) L2 at 16128..16847 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) gatenet_L3_Flash + 16128), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) gatenet_L2_Memory + 16128), 720, 0, UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
	/* Moving Modelquant_densebiasaddreadvar, size 12 from HyperFlash at 17776 to (size 12) L2 at 17776..17787 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) gatenet_L3_Flash + 17776), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) gatenet_L2_Memory + 17776), 12, 0, UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
	/* Moving S21_Mul_scale, size 3 from HyperFlash at 17788 to (size 3) L2 at 17788..17790 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) gatenet_L3_Flash + 17788), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) gatenet_L2_Memory + 17788), 3, 0, UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
	/* Moving S21_Mul_shift, size 3 from HyperFlash at 17792 to (size 3) L2 at 17792..17794 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) gatenet_L3_Flash + 17792), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) gatenet_L2_Memory + 17792), 3, 0, UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
	/* Moving S21_Infos, size 13 from HyperFlash at 17760 to (size 13) L2 at 17760..17772 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) gatenet_L3_Flash + 17760), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) gatenet_L2_Memory + 17760), 13, 0, UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
	return 0;
}
int gatenetCNN_Destruct()

{
	AT_L2_FREE(0, gatenet_L2_Memory_Dyn, 129600);
	AT_L2_FREE(0, gatenet_L2_Memory, 17796);
	AT_L1_FREE(0, gatenet_L1_Memory, 45136);
	AT_HYPERFLASH_FS_CLOSE(&HyperFlash);
	return 0;
}
int gatenetCNN_Memory(int Which)

{
	switch (Which) {
		case 0: return 45136;
		case 1: return 129600;
		case 2: return 17796;
	}
	return 0;
}
int gatenetCNN(
		signed char * __restrict__ Input_1,
		signed char * __restrict__ Output_1)

{
	S3_Conv2d_16x1x3x3_MaxPool_2x2_Relu(
		((signed char * __restrict__) Input_1), /* In */
		((signed char * __restrict__) (gatenet_L2_Memory+16848)), /* Filter */
		((signed int * __restrict__) (gatenet_L2_Memory+17120)), /* Bias */
		((signed char * __restrict__) (gatenet_L2_Memory_Dyn+43200)), /* Out */
		((unsigned char * __restrict__) (gatenet_L2_Memory+17504)), /* Scale */
		((signed char * __restrict__) (gatenet_L2_Memory+17520)), /* ScaleN */
		((signed char * __restrict__) (gatenet_L2_Memory+17536)) /* Infos */
	);
	S6_Conv2d_32x16x3x3_MaxPool_2x2_Relu(
		((signed char * __restrict__) (gatenet_L2_Memory_Dyn+43200)), /* In */
		((signed char * __restrict__) (gatenet_L2_Memory+0)), /* Filter */
		((signed int * __restrict__) (gatenet_L2_Memory+16992)), /* Bias */
		((signed char * __restrict__) (gatenet_L2_Memory_Dyn+0)), /* Out */
		((unsigned char * __restrict__) (gatenet_L2_Memory+17440)), /* Scale */
		((signed char * __restrict__) (gatenet_L2_Memory+17472)), /* ScaleN */
		((signed char * __restrict__) (gatenet_L2_Memory+17552)) /* Infos */
	);
	S9_Conv2d_16x32x3x3_MaxPool_2x2_Relu(
		((signed char * __restrict__) (gatenet_L2_Memory_Dyn+0)), /* In */
		((signed char * __restrict__) (gatenet_L2_Memory+4608)), /* Filter */
		((signed int * __restrict__) (gatenet_L2_Memory+17184)), /* Bias */
		((signed char * __restrict__) (gatenet_L2_Memory_Dyn+43200)), /* Out */
		((unsigned char * __restrict__) (gatenet_L2_Memory+17568)), /* Scale */
		((signed char * __restrict__) (gatenet_L2_Memory+17584)), /* ScaleN */
		((signed char * __restrict__) (gatenet_L2_Memory+17600)) /* Infos */
	);
	S12_Conv2d_16x16x3x3_MaxPool_2x2_Relu(
		((signed char * __restrict__) (gatenet_L2_Memory_Dyn+43200)), /* In */
		((signed char * __restrict__) (gatenet_L2_Memory+9216)), /* Filter */
		((signed int * __restrict__) (gatenet_L2_Memory+17248)), /* Bias */
		((signed char * __restrict__) (gatenet_L2_Memory_Dyn+0)), /* Out */
		((unsigned char * __restrict__) (gatenet_L2_Memory+17616)), /* Scale */
		((signed char * __restrict__) (gatenet_L2_Memory+17632)), /* ScaleN */
		((signed char * __restrict__) (gatenet_L2_Memory+17648)) /* Infos */
	);
	S15_Conv2d_16x16x3x3_MaxPool_2x2_Relu(
		((signed char * __restrict__) (gatenet_L2_Memory_Dyn+0)), /* In */
		((signed char * __restrict__) (gatenet_L2_Memory+11520)), /* Filter */
		((signed int * __restrict__) (gatenet_L2_Memory+17312)), /* Bias */
		((signed char * __restrict__) (gatenet_L2_Memory_Dyn+1232)), /* Out */
		((unsigned char * __restrict__) (gatenet_L2_Memory+17664)), /* Scale */
		((signed char * __restrict__) (gatenet_L2_Memory+17680)), /* ScaleN */
		((signed char * __restrict__) (gatenet_L2_Memory+17696)) /* Infos */
	);
	S18_Conv2d_16x16x3x3_Relu(
		((signed char * __restrict__) (gatenet_L2_Memory_Dyn+1232)), /* In */
		((signed char * __restrict__) (gatenet_L2_Memory+13824)), /* Filter */
		((signed int * __restrict__) (gatenet_L2_Memory+17376)), /* Bias */
		((signed char * __restrict__) (gatenet_L2_Memory_Dyn+0)), /* Out */
		((unsigned char * __restrict__) (gatenet_L2_Memory+17712)), /* Scale */
		((signed char * __restrict__) (gatenet_L2_Memory+17728)), /* ScaleN */
		((signed char * __restrict__) (gatenet_L2_Memory+17744)) /* Infos */
	);
	S21_Op_FULLY_CONNECTED_0_13_fusion(
		((signed char * __restrict__) (gatenet_L2_Memory_Dyn+0)), /* In */
		((signed char * __restrict__) (gatenet_L2_Memory+16128)), /* Filter */
		((signed int * __restrict__) (gatenet_L2_Memory+17776)), /* Bias */
		((signed char * __restrict__) Output_1), /* Out */
		((unsigned char * __restrict__) (gatenet_L2_Memory+17788)), /* Scale */
		((signed char * __restrict__) (gatenet_L2_Memory+17792)), /* ScaleN */
		((signed char * __restrict__) (gatenet_L2_Memory+17760)) /* Infos */
	);
	return 0;
}
