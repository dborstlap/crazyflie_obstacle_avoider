#ifndef CLASSIFICATION_GRAPHINFO_H
#define CLASSIFICATION_GRAPHINFO_H
// Quantized scales can be used round_norm(val * QSCALE, QNORM) giving the real value in Q8
// Input_1
#define classification_Input_1_OUT_SCALE	0.007874016213692783
#define classification_Input_1_OUT_QSCALE	65
#define classification_Input_1_OUT_QNORM	13
#define classification_Input_1_OUT_ZERO_POINT	0
// S1_Op_DEQUANTIZE_0_3
#define classification_S1_Op_DEQUANTIZE_0_3_OUT_SCALE	0.0015700362855568528
#define classification_S1_Op_DEQUANTIZE_0_3_OUT_QSCALE	103
#define classification_S1_Op_DEQUANTIZE_0_3_OUT_QNORM	16
#define classification_S1_Op_DEQUANTIZE_0_3_OUT_ZERO_POINT	0
// S2_Op_sequentialquant_conv2dBiasAddR
#define classification_S2_Op_sequentialquant_conv2dBiasAddR_OUT_SCALE	1.2362491168560652e-05
#define classification_S2_Op_sequentialquant_conv2dBiasAddR_OUT_QSCALE	104
#define classification_S2_Op_sequentialquant_conv2dBiasAddR_OUT_QNORM	23
#define classification_S2_Op_sequentialquant_conv2dBiasAddR_OUT_ZERO_POINT	0
// S3_Conv2d_16x1x3x3_MaxPool_2x2_Relu
#define classification_S3_Conv2d_16x1x3x3_MaxPool_2x2_Relu_OUT_SCALE	0.007874015718698502
#define classification_S3_Conv2d_16x1x3x3_MaxPool_2x2_Relu_OUT_QSCALE	65
#define classification_S3_Conv2d_16x1x3x3_MaxPool_2x2_Relu_OUT_QNORM	13
#define classification_S3_Conv2d_16x1x3x3_MaxPool_2x2_Relu_OUT_ZERO_POINT	0
// S4_Op_DEQUANTIZE_0_9
#define classification_S4_Op_DEQUANTIZE_0_9_OUT_SCALE	0.0011778463376685977
#define classification_S4_Op_DEQUANTIZE_0_9_OUT_QSCALE	77
#define classification_S4_Op_DEQUANTIZE_0_9_OUT_QNORM	16
#define classification_S4_Op_DEQUANTIZE_0_9_OUT_ZERO_POINT	0
// S5_Op_sequentialquant_conv2d_1BiasAd
#define classification_S5_Op_sequentialquant_conv2d_1BiasAd_OUT_SCALE	9.274380317947362e-06
#define classification_S5_Op_sequentialquant_conv2d_1BiasAd_OUT_QSCALE	78
#define classification_S5_Op_sequentialquant_conv2d_1BiasAd_OUT_QNORM	23
#define classification_S5_Op_sequentialquant_conv2d_1BiasAd_OUT_ZERO_POINT	0
// S6_Conv2d_32x16x3x3_MaxPool_2x2_Relu
#define classification_S6_Conv2d_32x16x3x3_MaxPool_2x2_Relu_OUT_SCALE	0.007874015718698502
#define classification_S6_Conv2d_32x16x3x3_MaxPool_2x2_Relu_OUT_QSCALE	65
#define classification_S6_Conv2d_32x16x3x3_MaxPool_2x2_Relu_OUT_QNORM	13
#define classification_S6_Conv2d_32x16x3x3_MaxPool_2x2_Relu_OUT_ZERO_POINT	0
// S7_Op_DEQUANTIZE_0_15
#define classification_S7_Op_DEQUANTIZE_0_15_OUT_SCALE	0.0007926932303234935
#define classification_S7_Op_DEQUANTIZE_0_15_OUT_QSCALE	104
#define classification_S7_Op_DEQUANTIZE_0_15_OUT_QNORM	17
#define classification_S7_Op_DEQUANTIZE_0_15_OUT_ZERO_POINT	0
// S8_Op_sequentialquant_conv2d_2BiasAd
#define classification_S8_Op_sequentialquant_conv2d_2BiasAd_OUT_SCALE	6.241678875085199e-06
#define classification_S8_Op_sequentialquant_conv2d_2BiasAd_OUT_QSCALE	105
#define classification_S8_Op_sequentialquant_conv2d_2BiasAd_OUT_QNORM	24
#define classification_S8_Op_sequentialquant_conv2d_2BiasAd_OUT_ZERO_POINT	0
// S9_Conv2d_64x32x3x3_MaxPool_2x2_Relu
#define classification_S9_Conv2d_64x32x3x3_MaxPool_2x2_Relu_OUT_SCALE	0.00392156862745098
#define classification_S9_Conv2d_64x32x3x3_MaxPool_2x2_Relu_OUT_QSCALE	64
#define classification_S9_Conv2d_64x32x3x3_MaxPool_2x2_Relu_OUT_QNORM	14
#define classification_S9_Conv2d_64x32x3x3_MaxPool_2x2_Relu_OUT_ZERO_POINT	-128
// S10_Op_DEQUANTIZE_0_22
#define classification_S10_Op_DEQUANTIZE_0_22_OUT_SCALE	0.0008410969167016447
#define classification_S10_Op_DEQUANTIZE_0_22_OUT_QSCALE	110
#define classification_S10_Op_DEQUANTIZE_0_22_OUT_QNORM	17
#define classification_S10_Op_DEQUANTIZE_0_22_OUT_ZERO_POINT	0
// S11_Op_sequentialquant_denseBiasAddRe
#define classification_S11_Op_sequentialquant_denseBiasAddRe_OUT_SCALE	3.29841928118292e-06
#define classification_S11_Op_sequentialquant_denseBiasAddRe_OUT_QSCALE	111
#define classification_S11_Op_sequentialquant_denseBiasAddRe_OUT_QNORM	25
#define classification_S11_Op_sequentialquant_denseBiasAddRe_OUT_ZERO_POINT	0
// S12_Op_FULLY_CONNECTED_0_23_fusion
#define classification_S12_Op_FULLY_CONNECTED_0_23_fusion_OUT_SCALE	0.007874015748031496
#define classification_S12_Op_FULLY_CONNECTED_0_23_fusion_OUT_QSCALE	65
#define classification_S12_Op_FULLY_CONNECTED_0_23_fusion_OUT_QNORM	13
#define classification_S12_Op_FULLY_CONNECTED_0_23_fusion_OUT_ZERO_POINT	0
// Output_1
#define classification_Output_1_OUT_SCALE	0.007874015748031496
#define classification_Output_1_OUT_QSCALE	65
#define classification_Output_1_OUT_QNORM	13
#define classification_Output_1_OUT_ZERO_POINT	0
#endif //CLASSIFICATION_GRAPHINFO_H