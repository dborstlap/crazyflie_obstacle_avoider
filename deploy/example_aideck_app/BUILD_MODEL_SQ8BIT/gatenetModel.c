#include <stdint.h>
#include <stdio.h>
#include "AutoTilerLib.h"
#include "CNN_Generators_SQ8.h"

#include "CNN_Copy_Generators.h"





void gatenetModel(unsigned int L1Memory, unsigned int L2Memory, unsigned int L3Memory, unsigned int L3Flash)
{
    KernelOper_T Cop = KOP_CONV;

    // SetKernelOpts(KER_OPT_NONE, KER_OPT_BUFFER_PROMOTE);
    SetSymbolDynamics();

    SetUsedFilesNames(0, 3, "Gap.h", "gatenet.h", "CNN_BasicKernels_SQ8.h");
    SetGeneratedFilesNames("gatenetKernels.c", "gatenetKernels.h");


    SetMemoryDeviceInfos(4,
        AT_MEM_L1, L1Memory, "gatenet_L1_Memory", 0, 0,
        AT_MEM_L2, L2Memory, "gatenet_L2_Memory", 0, 1,
        AT_MEM_L3_HRAM, L3Memory, "gatenet_L3_Memory", 0, 0,
        AT_MEM_L3_HFLASH, L3Flash, "gatenet_L3_Flash", "gatenet_L3_Flash_Const.dat", 0
    );

    LoadCNN_SQ8_Library();


    // generator for CONV_2D_0_1_fusion
    CNN_ConvolutionPoolAct_SQ8("S3_Conv2d_16x1x3x3_MaxPool_2x2_Relu", 0,
                               4, 1,
                               1, 16, 180, 120,
                               KOP_CONV, 3, 3, 1, 1, 1, 1, 1,
                               KOP_MAXPOOL, 2, 2, 1, 1, 2, 2, 0,
                               KOP_RELU);
    
    // generator for CONV_2D_0_3_fusion
    CNN_ConvolutionPoolAct_SQ8("S6_Conv2d_32x16x3x3_MaxPool_2x2_Relu", 0,
                               4, 1,
                               16, 32, 90, 60,
                               KOP_CONV, 3, 3, 1, 1, 1, 1, 1,
                               KOP_MAXPOOL, 2, 2, 1, 1, 2, 2, 0,
                               KOP_RELU);
    
    // generator for CONV_2D_0_5_fusion
    CNN_ConvolutionPoolAct_SQ8("S9_Conv2d_16x32x3x3_MaxPool_2x2_Relu", 0,
                               4, 1,
                               32, 16, 45, 30,
                               KOP_CONV, 3, 3, 1, 1, 1, 1, 1,
                               KOP_MAXPOOL, 2, 2, 1, 1, 2, 2, 0,
                               KOP_RELU);
    
    // generator for CONV_2D_0_7_fusion
    CNN_ConvolutionPoolAct_SQ8("S12_Conv2d_16x16x3x3_MaxPool_2x2_Relu", 0,
                               4, 1,
                               16, 16, 22, 15,
                               KOP_CONV, 3, 3, 1, 1, 1, 1, 1,
                               KOP_MAXPOOL, 2, 2, 1, 1, 2, 2, 0,
                               KOP_RELU);
    
    // generator for CONV_2D_0_9_fusion
    CNN_ConvolutionPoolAct_SQ8("S15_Conv2d_16x16x3x3_MaxPool_2x2_Relu", 0,
                               4, 1,
                               16, 16, 11, 7,
                               KOP_CONV, 3, 3, 1, 1, 1, 1, 1,
                               KOP_MAXPOOL, 2, 2, 1, 1, 2, 2, 0,
                               KOP_RELU);
    
    // generator for CONV_2D_0_11_fusion
    CNN_ConvolutionPoolAct_SQ8("S18_Conv2d_16x16x3x3_Relu", 0,
                               4, 1,
                               16, 16, 5, 3,
                               KOP_CONV, 3, 3, 1, 1, 1, 1, 1,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_RELUM);
    
    // generator for FULLY_CONNECTED_0_13_fusion
    CNN_LinearAct_SQ8("S21_Op_FULLY_CONNECTED_0_13_fusion", 0,
                      4, 1,
                      240, 3,
                      KOP_LINEAR, KOP_RELU);
    

#define GRAPH
#ifdef GRAPH
    CreateGraph("gatenetCNN",
        /* Arguments either passed or globals */
            CArgs(37,
                TCArgInfo("signed char * __restrict__", "Input_1", ARG_SCOPE_ARG, ARG_DIR_IN, AT_MEM_L2, AT_MEM_L2, 0),
                TCArgInfo("signed char * __restrict__", "Modelquant_conv1conv2dmodelqua", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Modelquant_conv1conv2dmodelqua.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Modelquant_activationrelumodel", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Modelquant_activationrelumodel.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S3_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S3_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S3_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S3_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.19556 out: 0.19556  actscale: [1] actscalen: [0] a0: [0] b0: 0 c0: 0 BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S3_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S3_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Modelquant_conv2conv2dmodelqua", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Modelquant_conv2conv2dmodelqua.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Modelquant_activation_1relumod", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Modelquant_activation_1relumod.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S6_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S6_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S6_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S6_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.12206 out: 0.12206  actscale: [1] actscalen: [0] a0: [0] b0: 0 c0: 0 BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S6_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S6_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Modelquant_conv3conv2dmodelqua", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Modelquant_conv3conv2dmodelqua.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Modelquant_activation_2relumod", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Modelquant_activation_2relumod.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S9_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S9_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S9_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S9_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.07567 out: 0.07567  actscale: [1] actscalen: [0] a0: [0] b0: 0 c0: 0 BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S9_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S9_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Modelquant_conv4conv2dmodelqua", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Modelquant_conv4conv2dmodelqua.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Modelquant_activation_3relumod", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Modelquant_activation_3relumod.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S12_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S12_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S12_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S12_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.05358 out: 0.05358  actscale: [1] actscalen: [0] a0: [0] b0: 0 c0: 0 BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S12_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S12_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Modelquant_conv5conv2dmodelqua", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Modelquant_conv5conv2dmodelqua.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Modelquant_activation_4relumod", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Modelquant_activation_4relumod.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S15_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S15_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S15_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S15_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.05385 out: 0.05385  actscale: [1] actscalen: [0] a0: [0] b0: 0 c0: 0 BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S15_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S15_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Modelquant_conv6conv2dmodelqua", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Modelquant_conv6conv2dmodelqua.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Modelquant_activation_5relumod", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Modelquant_activation_5relumod.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S18_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S18_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S18_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S18_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.03233 out: 0.03233  actscale: [1] actscalen: [0] a0: [-128] b0: 0 c0: 0 BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S18_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S18_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Modelquant_densematmulmodelqua", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Modelquant_densematmulmodelqua.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Modelquant_densebiasaddreadvar", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Modelquant_densebiasaddreadvar.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S21_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S21_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S21_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S21_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.84891 out: 0.84891  actscale: [1] actscalen: [0] a0: [0] b0: 0 c0: 0 BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S21_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S21_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Output_1", ARG_SCOPE_ARG, ARG_DIR_OUT, AT_MEM_L2, AT_MEM_L2, 0)
            ),
        /* Locals, allocated dynamically */
        CArgs(6,
            TCArgInfo("signed char * __restrict__", "S3_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S6_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S9_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S12_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S15_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S18_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0)
        )
    );

    /* Stacked tensors - Concats */
    // no concats in graph so not stacked tensors created

    // Node S3_Conv2d_16x1x3x3_MaxPool_2x2_Relu inq -257.01<(i8-0.00)*2.00787402<255.00 forced weightsq chan<(i8-0.00)*chan<chan outq -25.03<(i8-0.00)*0.19555800<24.84 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S3_Conv2d_16x1x3x3_MaxPool_2x2_Relu",
        Bindings(7,
            GNodeArg(GNA_IN, "Input_1", 0),
            GNodeArg(GNA_IN, "Modelquant_conv1conv2dmodelqua", 0),
            GNodeArg(GNA_IN, "Modelquant_activationrelumodel", 0),
            GNodeArg(GNA_OUT, "S3_Output", 0),
            GNodeArg(GNA_IN, "S3_Mul_scale", 0),
            GNodeArg(GNA_IN, "S3_Mul_shift", 0),
            GNodeArg(GNA_IN, "S3_Infos", 0)
        )
    );
    // Node S6_Conv2d_32x16x3x3_MaxPool_2x2_Relu inq -25.03<(i8-0.00)*0.19555800<24.84 forced weightsq chan<(i8-0.00)*chan<chan outq -15.62<(i8-0.00)*0.12205722<15.50 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S6_Conv2d_32x16x3x3_MaxPool_2x2_Relu",
        Bindings(7,
            GNodeArg(GNA_IN, "S3_Output", 0),
            GNodeArg(GNA_IN, "Modelquant_conv2conv2dmodelqua", 0),
            GNodeArg(GNA_IN, "Modelquant_activation_1relumod", 0),
            GNodeArg(GNA_OUT, "S6_Output", 0),
            GNodeArg(GNA_IN, "S6_Mul_scale", 0),
            GNodeArg(GNA_IN, "S6_Mul_shift", 0),
            GNodeArg(GNA_IN, "S6_Infos", 0)
        )
    );
    // Node S9_Conv2d_16x32x3x3_MaxPool_2x2_Relu inq -15.62<(i8-0.00)*0.12205722<15.50 forced weightsq chan<(i8-0.00)*chan<chan outq -9.69<(i8-0.00)*0.07567123<9.61 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S9_Conv2d_16x32x3x3_MaxPool_2x2_Relu",
        Bindings(7,
            GNodeArg(GNA_IN, "S6_Output", 0),
            GNodeArg(GNA_IN, "Modelquant_conv3conv2dmodelqua", 0),
            GNodeArg(GNA_IN, "Modelquant_activation_2relumod", 0),
            GNodeArg(GNA_OUT, "S9_Output", 0),
            GNodeArg(GNA_IN, "S9_Mul_scale", 0),
            GNodeArg(GNA_IN, "S9_Mul_shift", 0),
            GNodeArg(GNA_IN, "S9_Infos", 0)
        )
    );
    // Node S12_Conv2d_16x16x3x3_MaxPool_2x2_Relu inq -9.69<(i8-0.00)*0.07567123<9.61 forced weightsq chan<(i8-0.00)*chan<chan outq -6.86<(i8-0.00)*0.05358156<6.80 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S12_Conv2d_16x16x3x3_MaxPool_2x2_Relu",
        Bindings(7,
            GNodeArg(GNA_IN, "S9_Output", 0),
            GNodeArg(GNA_IN, "Modelquant_conv4conv2dmodelqua", 0),
            GNodeArg(GNA_IN, "Modelquant_activation_3relumod", 0),
            GNodeArg(GNA_OUT, "S12_Output", 0),
            GNodeArg(GNA_IN, "S12_Mul_scale", 0),
            GNodeArg(GNA_IN, "S12_Mul_shift", 0),
            GNodeArg(GNA_IN, "S12_Infos", 0)
        )
    );
    // Node S15_Conv2d_16x16x3x3_MaxPool_2x2_Relu inq -6.86<(i8-0.00)*0.05358156<6.80 forced weightsq chan<(i8-0.00)*chan<chan outq -6.89<(i8-0.00)*0.05385065<6.84 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S15_Conv2d_16x16x3x3_MaxPool_2x2_Relu",
        Bindings(7,
            GNodeArg(GNA_IN, "S12_Output", 0),
            GNodeArg(GNA_IN, "Modelquant_conv5conv2dmodelqua", 0),
            GNodeArg(GNA_IN, "Modelquant_activation_4relumod", 0),
            GNodeArg(GNA_OUT, "S15_Output", 0),
            GNodeArg(GNA_IN, "S15_Mul_scale", 0),
            GNodeArg(GNA_IN, "S15_Mul_shift", 0),
            GNodeArg(GNA_IN, "S15_Infos", 0)
        )
    );
    // Node S18_Conv2d_16x16x3x3_Relu inq -6.89<(i8-0.00)*0.05385065<6.84 forced weightsq chan<(i8-0.00)*chan<chan outq 0.00<(i8--128.00)*0.03232917<8.24 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S18_Conv2d_16x16x3x3_Relu",
        Bindings(7,
            GNodeArg(GNA_IN, "S15_Output", 0),
            GNodeArg(GNA_IN, "Modelquant_conv6conv2dmodelqua", 0),
            GNodeArg(GNA_IN, "Modelquant_activation_5relumod", 0),
            GNodeArg(GNA_OUT, "S18_Output", 0),
            GNodeArg(GNA_IN, "S18_Mul_scale", 0),
            GNodeArg(GNA_IN, "S18_Mul_shift", 0),
            GNodeArg(GNA_IN, "S18_Infos", 0)
        )
    );
    // Node FULLY_CONNECTED_0_13 inq 0.00<(i8--128.00)*0.03232917<8.24 weightsq chan<(i8-0.00)*chan<chan outq -108.66<(i8-0.00)*0.84890533<107.81 forced
    AddNode("S21_Op_FULLY_CONNECTED_0_13_fusion",
        Bindings(7,
            GNodeArg(GNA_IN, "S18_Output", 0),
            GNodeArg(GNA_IN, "Modelquant_densematmulmodelqua", 0),
            GNodeArg(GNA_IN, "Modelquant_densebiasaddreadvar", 0),
            GNodeArg(GNA_OUT, "Output_1", 0),
            GNodeArg(GNA_IN, "S21_Mul_scale", 0),
            GNodeArg(GNA_IN, "S21_Mul_shift", 0),
            GNodeArg(GNA_IN, "S21_Infos", 0)
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
    gatenetModel(64000, 300000, 8000000, 64*1024*1024);
    GenerateTilingCode();
    return 0;
}
