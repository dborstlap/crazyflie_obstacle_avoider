/**
 * ,---------,       ____  _ __
 * |  ,-^-,  |      / __ )(_) /_______________ _____  ___
 * | (  O  ) |     / __  / / __/ ___/ ___/ __ `/_  / / _ \
 * | / ,--´  |    / /_/ / / /_/ /__/ /  / /_/ / / /_/  __/
 *    +------`   /_____/_/\__/\___/_/   \__,_/ /___/\___/
 *
 * AI-deck examples
 *
 * Copyright (C) 2021 Bitcraze AB
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, in version 3.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 *
 * @file classification.c
 *
 *
 */

#include "gatenet.h"
#include "bsp/camera/himax.h"
#include "bsp/transport/nina_w10.h"
#include "gatenetKernels.h"
#include "gaplib/ImgIO.h"
#include "pmsis.h"
#include "stdio.h"
#include "bsp/bsp.h"
#include "cpx.h"
#include "bsp/ram.h"
#include "bsp/ram/hyperram.h"
#include "imageUtils.h"
#include "bsp/flash/hyperflash.h"
#include "img_proc.h"
#include "cameraParameters.h"


//For streaming
#ifdef USE_STREAMER
#include "wifi.h"
#include "bsp/buffer.h"

static pi_buffer_t buffer;
static pi_buffer_t buffer_out;
#endif
#define IMG_ORIENTATION 0x0101
#define CAM_WIDTH 162
#define CAM_HEIGHT 122
#define CHANNELS 1

#define NN_INPUT_WIDTH 180
#define NN_INPUT_HEIGHT 120
#define NN_INPUT_CHANNELS 1

typedef signed char NETWORK_OUT_TYPE;

static pi_task_t task1;
static unsigned char *imgBuff;
static unsigned char *inputNetwork;

L2_MEM NETWORK_OUT_TYPE *Output_1;
static float drone_pos[3]; // x, y, z
static struct pi_device camera;
static struct pi_device cluster_dev;
static struct pi_cluster_task *task;
static struct pi_cluster_conf cluster_conf;
struct pi_device device;
struct cameraParameters cameraIntrinsic;

//UART init param
L2_MEM struct pi_uart_conf uart_conf;
L2_MEM struct pi_device uart;
L2_MEM uint8_t rec_digit = -1;

static CPXPacket_t txpCrazyfly;


AT_HYPERFLASH_FS_EXT_ADDR_TYPE __PREFIX(_L3_Flash) = 0;

static EventGroupHandle_t evGroup;
#define CAPTURE_DONE_BIT (1 << 0)

// Performance menasuring variables
static uint32_t start = 0;
static uint32_t captureTime = 0;
static uint32_t resizeTime = 0;
static uint32_t inferenceTime = 0;
static uint32_t timetot = 0;
static uint32_t imagetime = 0;
static uint32_t positionTime = 0;
// #define OUTPUT_PROFILING_DATA

static int open_camera_himax(struct pi_device *device)
{
  struct pi_himax_conf cam_conf;

  pi_himax_conf_init(&cam_conf);

  cam_conf.format = PI_CAMERA_QQVGA;

  pi_open_from_conf(device, &cam_conf);
  if (pi_camera_open(device))
    return -1;

  // rotate image
  pi_camera_control(device, PI_CAMERA_CMD_START, 0);
  uint8_t set_value = 3;
  uint8_t reg_value;
  pi_camera_reg_set(device, IMG_ORIENTATION, &set_value);
  pi_time_wait_us(1000000);
  pi_camera_reg_get(device, IMG_ORIENTATION, &reg_value);
  if (set_value != reg_value)
  {
    cpxPrintToConsole(LOG_TO_CRTP, "Failed to rotate camera image\n");
    return -1;
  }
  pi_camera_control(device, PI_CAMERA_CMD_STOP, 0);
  pi_camera_control(device, PI_CAMERA_CMD_AEG_INIT, 0);

  return 0;
}

static int open_camera(struct pi_device *device)
{
  return open_camera_himax(device);
}

static void capture_done_cb(void *arg)
{
  xEventGroupSetBits(evGroup, CAPTURE_DONE_BIT);
}

#ifdef USE_STREAMER
static int wifiConnected = 0;
static int wifiClientConnected = 0;

static CPXPacket_t rxp;
void rx_task(void *parameters)
{
  while (1)
  {
    cpxReceivePacketBlocking(CPX_F_WIFI_CTRL, &rxp);

    WiFiCTRLPacket_t * wifiCtrl = (WiFiCTRLPacket_t*) rxp.data;

    switch (wifiCtrl->cmd)
    {
      case WIFI_CTRL_STATUS_WIFI_CONNECTED:
        cpxPrintToConsole(LOG_TO_CRTP, "Wifi connected (%u.%u.%u.%u)\n",
                          wifiCtrl->data[0], wifiCtrl->data[1],
                          wifiCtrl->data[2], wifiCtrl->data[3]);
        wifiConnected = 1;
        break;
      case WIFI_CTRL_STATUS_CLIENT_CONNECTED:
        cpxPrintToConsole(LOG_TO_CRTP, "Wifi client connection status: %u\n", wifiCtrl->data[0]);
        wifiClientConnected = wifiCtrl->data[0];
        break;
      default:
        break;
    }
  }
}



typedef struct
{
  uint8_t magic;
  uint16_t width;
  uint16_t height;
  uint8_t depth;
  uint8_t type;
  uint32_t size;
} __attribute__((packed)) img_header_t;


typedef enum
{
  RAW_ENCODING = 0,
  JPEG_ENCODING = 1
} __attribute__((packed)) StreamerMode_t;


static StreamerMode_t streamerMode = RAW_ENCODING;

static CPXPacket_t txp;

void createImageHeaderPacket(CPXPacket_t * packet, uint32_t imgSize, StreamerMode_t imgType) {
  img_header_t *imgHeader = (img_header_t *) packet->data;
  imgHeader->magic = 0xBC;
  imgHeader->width = NN_INPUT_WIDTH;
  imgHeader->height = NN_INPUT_HEIGHT;
  imgHeader->depth = imagetime;
  imgHeader->type = imgType;
  imgHeader->size = imgSize;
  packet->dataLength = sizeof(img_header_t);
}

void sendBufferViaCPX(CPXPacket_t * packet, uint8_t * buffer, uint32_t bufferSize) {
  uint32_t offset = 0;
  uint32_t size = 0;
  do {
    size = sizeof(packet->data);
    if (offset + size > bufferSize)
    {
      size = bufferSize - offset;
    }
    memcpy(packet->data, &buffer[offset], sizeof(packet->data));
    packet->dataLength = size;
    cpxSendPacketBlocking(packet);
    offset += size;
  } while (size == sizeof(packet->data));
}

#ifdef SETUP_WIFI_AP
void setupWiFi(void) {
  static char ssid[] = "WiFi streaming example";
  cpxPrintToConsole(LOG_TO_CRTP, "Setting up WiFi AP\n");
  // Set up the routing for the WiFi CTRL packets
  txp.route.destination = CPX_T_ESP32;
  rxp.route.source = CPX_T_GAP8;
  txp.route.function = CPX_F_WIFI_CTRL;
  WiFiCTRLPacket_t * wifiCtrl = (WiFiCTRLPacket_t*) txp.data;
  
  wifiCtrl->cmd = WIFI_CTRL_SET_SSID;
  memcpy(wifiCtrl->data, ssid, sizeof(ssid));
  txp.dataLength = sizeof(ssid);
  cpxSendPacketBlocking(&txp);
  
  wifiCtrl->cmd = WIFI_CTRL_WIFI_CONNECT;
  wifiCtrl->data[0] = 0x01;
  txp.dataLength = 2;
  cpxSendPacketBlocking(&txp);
}
#endif

#endif
static void RunNetwork()
{
  __PREFIX(CNN)
  (inputNetwork, Output_1);
}


void camera_task(void *parameters)
{

#ifdef SETUP_WIFI_AP
  setupWiFi();
#endif

  cpxPrintToConsole(LOG_TO_CRTP, "Starting camera task...\n");
  uint32_t resolution = CAM_WIDTH * CAM_HEIGHT * CHANNELS;
  uint32_t captureSize = resolution * sizeof(unsigned char);
  imgBuff = (unsigned char *)pmsis_l2_malloc(captureSize);
  if (imgBuff == NULL)
  {
    cpxPrintToConsole(LOG_TO_CRTP, "Failed to allocate Memory for Image \n");
    return;
  }

  inputNetwork = (unsigned char *)pmsis_l2_malloc(NN_INPUT_WIDTH*NN_INPUT_HEIGHT*NN_INPUT_CHANNELS* sizeof(unsigned char));
  if (inputNetwork == NULL)
  {
    cpxPrintToConsole(LOG_TO_CRTP, "Failed to allocate Memory for Network Input \n");
    return;
  }


  if (open_camera(&camera))
  {
    cpxPrintToConsole(LOG_TO_CRTP, "Failed to open camera\n");
    return;
  }

  cameraIntrinsic = getCameraParameters();

  //  UART init with Crazyflie and configure
  pi_uart_conf_init(&uart_conf);
  uart_conf.enable_tx = 1;
  uart_conf.enable_rx = 0;
  uart_conf.baudrate_bps = 115200;

  pi_open_from_conf(&uart, &uart_conf);
  if (pi_uart_open(&uart))
  {
    cpxPrintToConsole(LOG_TO_CRTP, "[UART] open failed !\n");
    pmsis_exit(-1);
  }
  cpxPrintToConsole(LOG_TO_CRTP, "[UART] Open\n");

  Output_1 = (NETWORK_OUT_TYPE *)pmsis_l2_malloc(3 * sizeof(NETWORK_OUT_TYPE));
  if (Output_1 == NULL)
  {
    cpxPrintToConsole(LOG_TO_CRTP, "Failed to allocate memory for output\n");
    pmsis_exit(-1);
  }

#ifdef USE_STREAMER
  buffer.data = inputNetwork;
  buffer.stride = 0; 
  pi_buffer_init(&buffer, PI_BUFFER_TYPE_L2, inputNetwork);
  pi_buffer_set_format(&buffer, NN_INPUT_WIDTH, NN_INPUT_HEIGHT, NN_INPUT_CHANNELS, PI_BUFFER_FORMAT_GRAY);
  pi_buffer_set_stride(&buffer, 0);
#endif
  // Configure Network task 
  /* Configure CNN task */
  pi_cluster_conf_init(&cluster_conf);
  pi_open_from_conf(&cluster_dev, (void *)&cluster_conf);
  if (pi_cluster_open(&cluster_dev))
  {
    printf("Cluster open failed !\n");
    pmsis_exit(-4);
  }
  pi_freq_set(PI_FREQ_DOMAIN_CL, FREQ_CL*1000*1000);
  task = pmsis_l2_malloc(sizeof(struct pi_cluster_task));
  if (!task)
  {
    cpxPrintToConsole(LOG_TO_CRTP, "failed to allocate memory for task\n");
  }
  cpxPrintToConsole(LOG_TO_CRTP, "Allocated memory for task\n");

  memset(task, 0, sizeof(struct pi_cluster_task));
  task->entry = &RunNetwork;
  task->stack_size = STACK_SIZE; // defined in makefile
  task->slave_stack_size = SLAVE_STACK_SIZE; // "
  task->arg = NULL;

  /* Construct CNN */
  int ret =  gatenetCNN_Construct();
  if (ret)
  {
    cpxPrintToConsole(LOG_TO_CRTP, "Failed to construct CNN with %d\n", ret);
    pmsis_exit(-5);
  }
  cpxPrintToConsole(LOG_TO_CRTP, "Constructed CNN\n");

  pi_camera_control(&camera, PI_CAMERA_CMD_STOP, 0);

#ifdef USE_STREAMER
  cpxInitRoute(CPX_T_GAP8, CPX_T_WIFI_HOST, CPX_F_APP, &txp.route);
#endif
  cpxInitRoute(CPX_T_GAP8, CPX_T_STM32, CPX_F_APP, &txpCrazyfly.route);


  uint32_t imgSize = 0;;
  pi_camera_control(&camera, PI_CAMERA_CMD_START, 0);
  while (1)
  {

    start = xTaskGetTickCount();

    pi_camera_capture(&camera, imgBuff, resolution);
    //xEventGroupWaitBits(evGroup, CAPTURE_DONE_BIT, pdTRUE, pdFALSE, (TickType_t)portMAX_DELAY);

    captureTime = xTaskGetTickCount() - start;
    start = xTaskGetTickCount();
    //cropImageFromCenter(imgBuff, CAM_WIDTH, CAM_HEIGHT, inputNetwork, NN_INPUT_WIDTH, NN_INPUT_HEIGHT);
    resizeGrayscaleImage(imgBuff, CAM_WIDTH, CAM_HEIGHT, inputNetwork, NN_INPUT_WIDTH, NN_INPUT_HEIGHT);
    resizeTime = xTaskGetTickCount() - start;
    imagetime = xTaskGetTickCount() - timetot;

    start = xTaskGetTickCount();
    pi_cluster_send_task_to_cl(&cluster_dev, task);
    inferenceTime = xTaskGetTickCount() - start;

    
    cpxPrintToConsole(LOG_TO_CRTP,"out0: %d, out1: %d, out2: %d\n", Output_1[0], Output_1[1],  Output_1[2]);
    
    //print all time
    //cpxPrintToConsole(LOG_TO_CRTP,"Inferece Time: %d, Capture Time: %d, Resize Time: %d, Postion Time: %d\n", inferenceTime, captureTime, resizeTime, positionTime);
#ifdef USE_STREAMER
    if (wifiClientConnected == 1)
    {
      //Draw On image
      drawRectangle(inputNetwork, NN_INPUT_WIDTH, NN_INPUT_HEIGHT, Output_1[0], Output_1[1],  Output_1[2], Output_1[3], Output_1[4], Output_1[5], Output_1[6], Output_1[7], 255);
      imgSize = NN_INPUT_HEIGHT * NN_INPUT_WIDTH * NN_INPUT_CHANNELS * sizeof(unsigned char);
      start = xTaskGetTickCount();

      // First send information about the image
      createImageHeaderPacket(&txp, captureSize, RAW_ENCODING);
      cpxSendPacketBlocking(&txp);

      start = xTaskGetTickCount();
      // Send image data
      // Make half image white
      sendBufferViaCPX(&txp, imgBuff, captureSize);

    }
#endif
  cpxPrintToConsole(LOG_TO_CRTP,"Drone Position: %f,%f,%f\n", drone_pos[0], drone_pos[1], drone_pos[2]);  
  txpCrazyfly.dataLength = sizeof(drone_pos);
  memcpy(txpCrazyfly.data,drone_pos,sizeof(drone_pos));
  cpxSendPacketBlocking(&txpCrazyfly);

  }
  pi_camera_control(&camera, PI_CAMERA_CMD_STOP, 0);
  __PREFIX(CNN_Destruct)();
}

#define LED_PIN 2
static pi_device_t led_gpio_dev;
void hb_task(void *parameters)
{
  (void)parameters;
  char *taskname = pcTaskGetName(NULL);

  // Initialize the LED pin
  pi_gpio_pin_configure(&led_gpio_dev, LED_PIN, PI_GPIO_OUTPUT);

  const TickType_t xDelay = 500 / portTICK_PERIOD_MS;

  while (1)
  {
    pi_gpio_pin_write(&led_gpio_dev, LED_PIN, 1);
    vTaskDelay(xDelay);
    pi_gpio_pin_write(&led_gpio_dev, LED_PIN, 0);
    vTaskDelay(xDelay);
  }
}

void start_tasks(void)
{

  cpxInit();
  cpxEnableFunction(CPX_F_WIFI_CTRL);

  cpxPrintToConsole(LOG_TO_CRTP, "-- GateNet --\n");

  evGroup = xEventGroupCreate();

  BaseType_t xTask;

  xTask = xTaskCreate(hb_task, "hb_task", configMINIMAL_STACK_SIZE * 2,
                      NULL, tskIDLE_PRIORITY + 1, NULL);
  if (xTask != pdPASS)
  {
    cpxPrintToConsole(LOG_TO_CRTP, "HB task did not start !\n");
    pmsis_exit(-1);
  }

  xTask = xTaskCreate(camera_task, "camera_task", configMINIMAL_STACK_SIZE * 4,
                      NULL, tskIDLE_PRIORITY + 1, NULL);

  if (xTask != pdPASS)
  {
    cpxPrintToConsole(LOG_TO_CRTP, "Camera task did not start !\n");
    pmsis_exit(-1);
  }
#ifdef USE_STREAMER
  xTask = xTaskCreate(rx_task, "rx_task", configMINIMAL_STACK_SIZE * 2,
                      NULL, tskIDLE_PRIORITY + 1, NULL);

  if (xTask != pdPASS)
  {
    cpxPrintToConsole(LOG_TO_CRTP, "RX task did not start !\n");
    pmsis_exit(-1);
  }
#endif

  while (1)
  {
    pi_yield();
  }
}

int main(void)
{
  pi_bsp_init();
  timetot = xTaskGetTickCount();
  // Increase the FC freq to 250 MHz
  pi_freq_set(PI_FREQ_DOMAIN_FC, 250000000);
  pi_pmu_voltage_set(PI_PMU_DOMAIN_FC, 1200);

  return pmsis_kickoff((void *)start_tasks);
}
