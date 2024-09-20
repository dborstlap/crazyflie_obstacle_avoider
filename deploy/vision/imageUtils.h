

#ifndef _IMAGE_UTILS_H_
#define _IMAGE_UTILS_H_


void downsampleBayer(const unsigned char* input, int inputWidth, int inputHeight, unsigned char* output, int outputWidth, int outputHeight);

void cropImageFromCenter(const unsigned char* input, int inputWidth, int inputHeight, unsigned char* output, int outputWidth, int outputHeight);

void drawRectangle( unsigned char* Img, int W, int H, int corner1x, int corner1y, int corner2x, int corner2y, int corner3x, int corner3y, int corner4x, int corner4y,int Value);

void resizeGrayscaleImage(const unsigned char* input_image, int input_width, int input_height, unsigned char* output_image, int output_width, int output_height);

#endif