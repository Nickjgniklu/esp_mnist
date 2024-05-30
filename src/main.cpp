#include <Arduino.h>
#include <Wire.h>
#include <SPI.h>
#include <FS.h>
#include <SPIFFS.h>
#include <WiFiClient.h>
#include <WiFi.h>
#include <WebServer.h>

#include "fb_gfx.h"
#include "main_functions.h"
#include "image_provider.h"
#include "model_settings.h"
#include <mnist_model.h>
#include <all_ops_resolver.h>

#include "tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "wifikeys.h"
#include "camera_config.h"
#include <ImageFormater.h>
#include <ConversionTools.h>
#define TAG "main"

// image size of images from the camera
size_t input_width = 320;
size_t input_height = 240;
size_t raw_image_size = input_width * input_height;

// image size of images accepted by the model
size_t model_input_height = 28;
size_t model_input_width = 28;
size_t model_input_size = model_input_height * model_input_width; // minst images are 28x28

/// @brief raw gray scale image
uint8_t *grayScaleBuffer;
/// @brief raw rgb image
uint8_t *rgbBuffer;
/// @brief jpeg image
uint8_t *jpegBuffer;
size_t jpegSize;

/// @brief wrapper object used for passing the grayScaleBuffer to the frame2jpg function
camera_fb_t *grayScalefb = new camera_fb_t();

OV2640 camera;
/// @brief used to create mnist like images
ImageFormater formatter;

// only allow these variables to be accessed in this file
namespace
{
  tflite::ErrorReporter *error_reporter = nullptr;
  const tflite::Model *model = nullptr;
  tflite::MicroInterpreter *interpreter = nullptr;
  TfLiteTensor *input = nullptr;
  // An area of memory to use for input, output, and intermediate arrays.
  const int kTensorArenaSize = 35 * 1024;
  static uint8_t tensor_arena[kTensorArenaSize];
}

/// @brief Writes the jpeg bytes to the serial port as binary
/// use JpegFilter to extract the jpeg bytes and save them to a file
/// @param jpegBuffer // the jpeg bytes to write
/// @param jpegSize // the length of the jpeg bytes
void serialWriteJpeg(uint8_t *jpegBuffer, size_t jpegSize)
{
  Serial.print("StartJPEG123456");
  Serial.write(jpegBuffer, jpegSize);
  Serial.print("EndJPEG123456");
}

/// @brief Updates the jpeg buffer with the current frame
/// from the grayScaleBuffer
void updateJpegBuffer()
{
  ESP_LOGI(TAG, "update jpeg buffer");

  free(jpegBuffer); // free the previous buffer if any
                    // frame2jpg will malloc the buffer for jpegBuffer
  frame2jpg(grayScalefb, 50, &jpegBuffer, &jpegSize);
  serialWriteJpeg(jpegBuffer, jpegSize);
  ESP_LOGI(TAG, "updated jpeg buffer");
}
/// @brief Returns the index of the max value
uint oneHotDecode(TfLiteTensor *layer)
{
  int max = 0;
  uint result = 0;
  for (uint i = 0; i < 10; i++)
  {
    // error_reporter->Report("num:%d score:%d", i,
    //                     output->data.int8[i]);
    if (layer->data.int8[i] > max)
    {
      result = i;
      max = layer->data.int8[i];
    }
  }
  return result;
}
/// @brief take as 28 *28 image and runs inference to predict a number
/// @param mnistimage
/// @return
uint inferMnistImage(int8_t *mnistimage)
{
  ESP_LOGI(TAG, "Start inference");

  memcpy(input->data.int8, mnistimage, model_input_size);

  int start = millis();
  ESP_LOGI(TAG, "Invoke");

  if (kTfLiteOk != interpreter->Invoke()) // Any error i have in invoke tend to just crash the whole system so i dont usually see this message
  {
    ESP_LOGE(TAG, "Invoke failed!");
  }
  else
  {
    ESP_LOGI(TAG, "Invoke success");
    ESP_LOGI(TAG, "Time taken: %d milliseconds", millis() - start);
  }

  TfLiteTensor *output = interpreter->output(0);
  uint result = oneHotDecode(output);
  return result;
}

/// @brief Draws a hollow rectangle on the frame buffer
void fb_gfx_drawRect(fb_data_t *fb, int32_t x, int32_t y, int32_t w, int32_t h, uint32_t color)
{
  fb_gfx_drawFastHLine(fb, x, y, w, color);
  fb_gfx_drawFastHLine(fb, x, y + h, w, color);
  fb_gfx_drawFastVLine(fb, x, y, h, color);
  fb_gfx_drawFastVLine(fb, x + w, y, h, color);
}

/// @brief debug function to print memory info
void print_memory_info()
{
  uint32_t free_heap = esp_get_free_heap_size();
  uint32_t total_psram = heap_caps_get_total_size(MALLOC_CAP_SPIRAM);
  uint32_t free_psram = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);

  ESP_LOGI("MemoryInfo", "Free internal heap: %u", free_heap);
  ESP_LOGI("MemoryInfo", "Total PSRAM: %u", total_psram);
  ESP_LOGI("MemoryInfo", "Free PSRAM: %u", free_psram);
}
void drawResult(size_t result)
{
  fb_data_t fbdata;
  fbdata.data = grayScaleBuffer;
  fbdata.width = input_width;
  fbdata.height = input_height;
  fbdata.format = FB_GRAY;
  fbdata.bytes_per_pixel = 1;
  char resultString[20];
  sprintf(resultString, "Result: %d", result);
  fb_gfx_print(&fbdata, input_width / 4, input_height / 4, 127, resultString);
}
WebServer server(80);
const char HEADER[] = "HTTP/1.1 200 OK\r\n"
                      "Access-Control-Allow-Origin: *\r\n"
                      "Content-Type: multipart/x-mixed-replace; boundary=123456789000000000000987654321\r\n";
const char BOUNDARY[] = "\r\n--123456789000000000000987654321\r\n";
const char CTNTTYPE[] = "Content-Type: image/jpeg\r\nContent-Length: ";
const int hdrLen = strlen(HEADER);
const int bdrLen = strlen(BOUNDARY);
const int cntLen = strlen(CTNTTYPE);

/// @brief Handles attempts at undefined routes
void handleNotFound()
{
  String message = "Server is running!\n\n";
  message += "URI: ";
  message += server.uri();
  message += "\nMethod: ";
  message += (server.method() == HTTP_GET) ? "GET" : "POST";
  message += "\nArguments: ";
  message += server.args();
  message += "\n";
  server.send(200, "text / plain", message);
}

/// @brief called when a client connects to the mjpeg stream
void handle_jpg_stream(void)
{
  ESP_LOGE(TAG, "handle_jpg_stream");

  char buf[32];
  int s;

  WiFiClient client = server.client();

  client.write(HEADER, hdrLen);
  client.write(BOUNDARY, bdrLen);

  while (true)
  {

    if (!client.connected())
    {

      ESP_LOGE(TAG, "Client disconnected");
      break;
    }

    print_memory_info();

    ESP_LOGI(TAG, "Load Camera Frame");
    camera.run();
    ESP_LOGI(TAG, "Got Camera Frame");

    fmt2rgb888(camera.getfb(), camera.getSize(), PIXFORMAT_JPEG, rgbBuffer);
    ConversionTools::uint8_rgb_to_uint8_grayscale(rgbBuffer, grayScaleBuffer, raw_image_size);

    // copy frame
    // memcpy(grayScaleBuffer, camera.getfb(), grayScalefb->width * grayScalefb->height);
    ESP_LOGI(TAG, "Copy Camera Frame");

    // convert to uint to int
    // should do this inplace
    ESP_LOGI(TAG, "Conversion to int8");

    int8_t *raw = (int8_t *)ps_malloc(grayScalefb->height * grayScalefb->width * sizeof(int8_t));

    ConversionTools::uint8_to_int8(grayScaleBuffer, raw, grayScalefb->width * grayScalefb->height);
    int8_t *mnist = (int8_t *)ps_malloc(28 * 28 * sizeof(int8_t));

    // preprocess image into minst format
    ESP_LOGI(TAG, "Create Mnist Image style from camera image");

    formatter.CreateMnistImageFromImage(raw, grayScalefb->width, grayScalefb->height, mnist);
    free(raw);
    ESP_LOGI(TAG, "overlay  mnist image");

    // put a mnist formatted image in the top left
    for (size_t i = 0; i < model_input_height; i++)
    {
      // copy the first 28 cols of each row in the mnist
      // to to corresponding row in the grayScaleBuffer
      ConversionTools::int8_to_uint8(mnist + ((model_input_width * i)), grayScaleBuffer + ((grayScalefb->width * i)), model_input_width);
    }

    uint result = inferMnistImage(mnist);
    ESP_LOGI(TAG, "update image with results %d", result);

    free(mnist);

    drawResult(result);

    updateJpegBuffer();

    client.write(CTNTTYPE, cntLen);
    sprintf(buf, "%d\r\n\r\n", jpegSize);
    client.write(buf, strlen(buf));
    client.write((char *)jpegBuffer, jpegSize);
    client.write(BOUNDARY, bdrLen);

    ESP_LOGI(TAG, "end frame");
  }
}

void initSerial()
{
  Serial.setRxBufferSize(1024);
  Serial.begin(115200);
  Serial.setTimeout(10000);
}
void printModelInfo()
{
  input = interpreter->input(0);
  ESP_LOGI(TAG, "Input Shape");
  for (int i = 0; i < input->dims->size; i++)
  {
    ESP_LOGI(TAG, "%d", input->dims->data[i]);
  }

  ESP_LOGI(TAG, "Input Type: %s", TfLiteTypeGetName(input->type));
  ESP_LOGI(TAG, "Output Shape");

  TfLiteTensor *output = interpreter->output(0);
  for (int i = 0; i < output->dims->size; i++)
  {
    ESP_LOGI(TAG, "%d", output->dims->data[i]);
  }
  ESP_LOGI(TAG, "Output Type: %s", TfLiteTypeGetName(output->type));

  ESP_LOGI(TAG, "Arena Size:%d bytes of memory", interpreter->arena_used_bytes());
}
void initTFInterpreter()
{
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;
  // Create Model
  model = tflite::GetModel(mnist_model);
  // Verify Version of Tf Micro matches Model's verson
  if (model->version() != TFLITE_SCHEMA_VERSION)
  {
    ESP_LOGE(TAG, "Model provided is schema version %d not equal to supported version %d.", model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }
  CREATE_ALL_OPS_RESOLVER(op_resolver)
  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(
      model, op_resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk)
  {
    ESP_LOGE(TAG, "AllocateTensors() failed");
    return;
  }

  printModelInfo();
}

void setup()
{
  delay(10000);
  rgbBuffer = (uint8_t *)ps_malloc(raw_image_size * 3);
  grayScaleBuffer = (uint8_t *)ps_malloc(raw_image_size);

  grayScalefb->buf = grayScaleBuffer;
  grayScalefb->format = PIXFORMAT_GRAYSCALE;
  grayScalefb->len = 0;
  grayScalefb->timestamp = {0, 0};
  grayScalefb->height = input_height;
  grayScalefb->width = input_width;

  initSerial();
  initTFInterpreter();
  IPAddress ip;
  ESP_LOGI(TAG, "Starting Camera");
  camera.init(esp32cam_ESPCam_config);
  ESP_LOGI(TAG, "Started Camera");

  ESP_LOGI(TAG, "Joining %s", ssid);
  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);
  delay(10000);

  while (WiFi.status() != WL_CONNECTED)
  {
    delay(10000);

    ESP_LOGI(TAG, "Waiting for connection to %s", ssid);
  }

  ip = WiFi.localIP();
  ESP_LOGI(TAG, "Connected with ip of %s", ip.toString().c_str());
  ESP_LOGI(TAG, "Starting mjpeg server at %s/mjpeg/1", ip.toString().c_str());
  server.on("/mjpeg/1", HTTP_GET, handle_jpg_stream);
  server.onNotFound(handleNotFound);
  server.begin();

  // fill the buffer with a black and white pattern
  for (int i = 0; i < raw_image_size; i++)
  {
    // set every other pixel to black and white
    grayScaleBuffer[i] = (i % 2) ? 0 : 255;
  }
  updateJpegBuffer();
}

void loop()
{
  server.handleClient();
  delay(10);
}