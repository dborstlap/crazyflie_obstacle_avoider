#include <math.h>
#include "imageUtils.h"
#include <stdio.h>
#include "cpx.h"


// Function to downsample a Bayer image
void downsampleBayer(const unsigned char* input, int inputWidth, int inputHeight, unsigned char* output, int outputWidth, int outputHeight) {
    float xRatio = (float)inputWidth / outputWidth;
    float yRatio = (float)inputHeight / outputHeight;
    int outputIndex = 0;

    for (int y = 0; y < outputHeight; y++) {
        for (int x = 0; x < outputWidth; x++) {
            int inputX = (int) round(x * xRatio);
            int inputY = (int) round(y * yRatio);
            int inputIndex = inputY * inputWidth + inputX;
            output[outputIndex] = input[inputIndex];
            outputIndex++;
        }
    }
}

void cropImageFromCenter(const unsigned char* input, int inputWidth, int inputHeight, unsigned char* output, int outputWidth, int outputHeight) {
    int startX = (inputWidth - outputWidth) / 2;
    int startY = (inputHeight - outputHeight) / 2;

    for (int y = 0; y < outputHeight; y++) {
        int inputOffset = (startY + y) * inputWidth + startX;
        int outputOffset = y * outputWidth;
        memcpy(output + outputOffset, input + inputOffset, outputWidth);
    }
}

void resizeGrayscaleImage(const unsigned char* input_image, int input_width, int input_height,
                          unsigned char* output_image, int output_width, int output_height) {

    float x_ratio = (float)(input_width - 1) / (output_width - 1);
    float y_ratio = (float)(input_height - 1) / (output_height - 1);

    for (int y = 0; y < output_height; y++) {
        float y_original = y * y_ratio;
        int y0 = (int)y_original;
        int y1 = y0 + 1;
        float y_fraction = y_original - y0;
        float y_frac_complement = 1.0 - y_fraction;

        for (int x = 0; x < output_width; x++) {
            float x_original = x * x_ratio;
            int x0 = (int)x_original;
            int x1 = x0 + 1;
            float x_fraction = x_original - x0;
            float x_frac_complement = 1.0 - x_fraction;

            // Bilinear interpolation
            char p00 = input_image[y0 * input_width + x0];
            char p01 = input_image[y0 * input_width + x1];
            char p10 = input_image[y1 * input_width + x0];
            char p11 = input_image[y1 * input_width + x1];

            float value = 
                p00 * (x_frac_complement) * (y_frac_complement) +
                p01 * (x_fraction) * (y_frac_complement) +
                p10 * (x_frac_complement) * (y_fraction) +
                p11 * (x_fraction) * (y_fraction);

            output_image[y * output_width + x] = (char)value;
        }
    }
}


void drawRectangle( unsigned char* Img, int W, int H, int corner1x, int corner1y, int corner2x, int corner2y, int corner3x, int corner3y, int corner4x, int corner4y,int Value)

{
    if (corner1x < 0 || corner1x >= W || corner1y < 0 || corner1y >= H ||
        corner2x < 0 || corner2x >= W || corner2y < 0 || corner2y >= H ||
        corner3x < 0 || corner3x >= W || corner3y < 0 || corner3y >= H ||
        corner4x < 0 || corner4x >= W || corner4y < 0 || corner4y >= H) {
        // Corners are outside the image, return without drawing anything
        return;
    }

    // Draw the rectangle by setting pixel values
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            if ((x == corner1x && y == corner1y) || (x == corner2x && y == corner2y) ||
                (x == corner3x && y == corner3y) || (x == corner4x && y == corner4y)) {
                // Draw a 4 by 4 rectangle around this point
                for (int dy = -2; dy <= 2; dy++) {
                    for (int dx = -2; dx <= 2; dx++) {
                        int x2 = x + dx;
                        int y2 = y + dy;
                        if (x2 >= 0 && x2 < W && y2 >= 0 && y2 < H) {
                            Img[y2 * W + x2] = Value;
                        }
                    }
                }
            }
        }
    }

}

