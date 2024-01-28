#include <stdint.h>
#include <stdio.h>
#include "AutoTilerLib.h"
#include "CNN_Generators_SQ8.h"

#include "CNN_Copy_Generators.h"





void classificationModel(unsigned int L1Memory, unsigned int L2Memory, unsigned int L3Memory, unsigned int L3Flash)
{
    KernelOper_T Cop = KOP_CONV;

    // SetKernelOpts(KER_OPT_NONE, KER_OPT_BUFFER_PROMOTE);
    SetSymbolDynamics();

    SetUsedFilesNames(0, 2, "CNN_BasicKernels_SQ8.h", "classification.h");
    SetGeneratedFilesNames("classificationKernels.c", "classificationKernels.h");


    SetMemoryDeviceInfos(4,
        AT_MEM_L1, L1Memory, "classification_L1_Memory", 0, 0,
        AT_MEM_L2, L2Memory, "classification_L2_Memory", 0, 1,
        AT_MEM_L3_HRAM, L3Memory, "classification_L3_Memory", 0, 0,
        AT_MEM_L3_HFLASH, L3Flash, "classification_L3_Flash", "classification_L3_Flash_Const.dat", 0
    );

    LoadCNN_SQ8_Library();


    // generator for CONV_2D_0_4_fusion
    CNN_ConvolutionPoolAct_SQ8("S3_Conv2d_16x1x3x3_MaxPool_2x2_Relu", 0, 4, 1,
                               1, 16, 324, 244,
                               KOP_CONV, 3, 3, 1, 1, 1, 1, 1,
                               KOP_MAXPOOL, 2, 2, 1, 1, 2, 2, 0,
                               KOP_RELU);
    
    // generator for CONV_2D_0_10_fusion
    CNN_ConvolutionPoolAct_SQ8("S6_Conv2d_32x16x3x3_MaxPool_2x2_Relu", 0, 4, 1,
                               16, 32, 162, 122,
                               KOP_CONV, 3, 3, 1, 1, 1, 1, 1,
                               KOP_MAXPOOL, 2, 2, 1, 1, 2, 2, 0,
                               KOP_RELU);
    
    // generator for CONV_2D_0_16_fusion
    CNN_ConvolutionPoolAct_SQ8("S9_Conv2d_64x32x3x3_MaxPool_2x2_Relu", 0, 4, 1,
                               32, 64, 81, 61,
                               KOP_CONV, 3, 3, 1, 1, 1, 1, 1,
                               KOP_MAXPOOL, 2, 2, 1, 1, 2, 2, 0,
                               KOP_RELU);
    
    // generator for FULLY_CONNECTED_0_23_fusion
    CNN_LinearAct_SQ8("S12_Op_FULLY_CONNECTED_0_23_fusion", 0,
                      4, 1,
                      76800, 1,
                      KOP_LINEAR, KOP_RELU);
    

#define GRAPH
#ifdef GRAPH
    CreateGraph("classificationCNN",
        /* Arguments either passed or globals */
            CArgs(22,
                TCArgInfo("signed char * __restrict__", "Input_1", ARG_SCOPE_ARG, ARG_DIR_IN, AT_MEM_L2, AT_MEM_L2, 0),
                TCArgInfo("signed char * __restrict__", "Dequantize_0_3", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Dequantize_0_3.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Sequentialquant_conv2dbiasaddr", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Sequentialquant_conv2dbiasaddr.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S3_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S3_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S3_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S3_Mul_shift.tensor", 1, 1, 8, 0)),
                // BiasQ: 0all 0
                TCArgInfo("signed char * __restrict__", "S3_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S3_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Dequantize_0_9", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Dequantize_0_9.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Sequentialquant_conv2d_1biasad", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Sequentialquant_conv2d_1biasad.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S6_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S6_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S6_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S6_Mul_shift.tensor", 1, 1, 8, 0)),
                // BiasQ: 0all 0
                TCArgInfo("signed char * __restrict__", "S6_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S6_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Dequantize_0_15", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Dequantize_0_15.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Sequentialquant_conv2d_2biasad", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Sequentialquant_conv2d_2biasad.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S9_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S9_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S9_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S9_Mul_shift.tensor", 1, 1, 8, 0)),
                // BiasQ: 0all 0
                TCArgInfo("signed char * __restrict__", "S9_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S9_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Dequantize_0_22", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Dequantize_0_22.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Sequentialquant_densebiasaddre", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Sequentialquant_densebiasaddre.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S12_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S12_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S12_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S12_Mul_shift.tensor", 1, 1, 8, 0)),
                // BiasQ: 0all 0
                TCArgInfo("signed char * __restrict__", "S12_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S12_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Output_1", ARG_SCOPE_ARG, ARG_DIR_OUT, AT_MEM_L2, AT_MEM_L2, 0)
            ),
        /* Locals, allocated dynamically */
        CArgs(3,
            TCArgInfo("signed char * __restrict__", "S3_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S6_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S9_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0)
        )
    );

    /* Stacked tensors - Concats */
    // no concats in graph so not stacked tensors created

    // Node S3_Conv2d_16x1x3x3_MaxPool_2x2_Relu inq -1.01<(i8-0.00)*0.00787402<1.00 forced weightsq chan<(i8-0.00)*chan<chan outq -1.01<(i8-0.00)*0.00787402<1.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S3_Conv2d_16x1x3x3_MaxPool_2x2_Relu",
        Bindings(7,
            GNodeArg(GNA_IN, "Input_1", 0),
            GNodeArg(GNA_IN, "Dequantize_0_3", 0),
            GNodeArg(GNA_IN, "Sequentialquant_conv2dbiasaddr", 0),
            GNodeArg(GNA_OUT, "S3_Output", 0),
            GNodeArg(GNA_IN, "S3_Mul_scale", 0),
            GNodeArg(GNA_IN, "S3_Mul_shift", 0),
            GNodeArg(GNA_IN, "S3_Infos", 0)
        )
    );
    // Node S6_Conv2d_32x16x3x3_MaxPool_2x2_Relu inq -1.01<(i8-0.00)*0.00787402<1.00 forced weightsq chan<(i8-0.00)*chan<chan outq -1.01<(i8-0.00)*0.00787402<1.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S6_Conv2d_32x16x3x3_MaxPool_2x2_Relu",
        Bindings(7,
            GNodeArg(GNA_IN, "S3_Output", 0),
            GNodeArg(GNA_IN, "Dequantize_0_9", 0),
            GNodeArg(GNA_IN, "Sequentialquant_conv2d_1biasad", 0),
            GNodeArg(GNA_OUT, "S6_Output", 0),
            GNodeArg(GNA_IN, "S6_Mul_scale", 0),
            GNodeArg(GNA_IN, "S6_Mul_shift", 0),
            GNodeArg(GNA_IN, "S6_Infos", 0)
        )
    );
    // Node S9_Conv2d_64x32x3x3_MaxPool_2x2_Relu inq -1.01<(i8-0.00)*0.00787402<1.00 forced weightsq chan<(i8-0.00)*chan<chan outq -1.01<(i8-0.00)*0.00787402<1.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S9_Conv2d_64x32x3x3_MaxPool_2x2_Relu",
        Bindings(7,
            GNodeArg(GNA_IN, "S6_Output", 0),
            GNodeArg(GNA_IN, "Dequantize_0_15", 0),
            GNodeArg(GNA_IN, "Sequentialquant_conv2d_2biasad", 0),
            GNodeArg(GNA_OUT, "S9_Output", 0),
            GNodeArg(GNA_IN, "S9_Mul_scale", 0),
            GNodeArg(GNA_IN, "S9_Mul_shift", 0),
            GNodeArg(GNA_IN, "S9_Infos", 0)
        )
    );
    // Node FULLY_CONNECTED_0_23 inq -1.01<(i8-0.00)*0.00787402<1.00 weightsq -0.11<(i8-0.00)*0.00084110<0.11 outq -1.01<(i8-0.00)*0.00787402<1.00
    AddNode("S12_Op_FULLY_CONNECTED_0_23_fusion",
        Bindings(7,
            GNodeArg(GNA_IN, "S9_Output", 0),
            GNodeArg(GNA_IN, "Dequantize_0_22", 0),
            GNodeArg(GNA_IN, "Sequentialquant_densebiasaddre", 0),
            GNodeArg(GNA_OUT, "Output_1", 0),
            GNodeArg(GNA_IN, "S12_Mul_scale", 0),
            GNodeArg(GNA_IN, "S12_Mul_shift", 0),
            GNodeArg(GNA_IN, "S12_Infos", 0)
        )
    );
    CloseGraph();
#endif
}

int main(int argc, char **argv)

{
    if (TilerParseOptions(argc, argv)) {
            printf("Failed to initialize or incorrect output arguments directory.\n"); return 1;
    }
    classificationModel(64000, 300000, 8000000, 20*1024*1024);
    GenerateTilingCode();
    return 0;
}
