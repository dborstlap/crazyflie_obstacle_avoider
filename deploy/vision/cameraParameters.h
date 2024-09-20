#ifndef _CAMERA_PARAMETERS_H_
#define _CAMERA_PARAMETERS_H_

struct cameraParameters
{
    /* data */
    float focal_x;
    float focal_y;
    float center_x;
    float center_y;
};
struct cameraParameters getCameraParameters();

struct cameraParameters getCameraParameters(){
    struct cameraParameters camera;
    camera.focal_x = 183.7350;
    camera.focal_y = 184.1241;
    camera.center_x = 166.9065;
    camera.center_y = 77.5140;
    return camera;
}
#endif