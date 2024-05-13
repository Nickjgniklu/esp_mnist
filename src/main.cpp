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
#include "mnist_model.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "camera_config.h"
#include "sampleDigits/sampleDigits.h"
#include "mbedtls/base64.h"
#include "ImageFormater.h"

// #define SOFTAP_MODE
#define ENABLE_MJPEG
#define TAG "main"
uint8_t *grayScaleBuffer;
uint8_t *jpegBytes;
size_t jpegSize;
size_t raw_image_size = (320 * 240);
camera_fb_t *grayScalefb = new camera_fb_t();
OV2640 camera;
ImageFormater formatter;
namespace
{
  tflite::ErrorReporter *error_reporter = nullptr;
  const tflite::Model *model = nullptr;
  tflite::MicroInterpreter *interpreter = nullptr;
  TfLiteTensor *input = nullptr;
  // An area of memory to use for input, output, and intermediate arrays.
  const int kTensorArenaSize = 35 * 1024;
  static uint8_t tensor_arena[kTensorArenaSize];
  using AllOpsResolver = tflite::MicroMutableOpResolver<128>;

  TfLiteStatus RegisterOps(AllOpsResolver &resolver)
  {
    // Not all of these are needed for every model but it can be know you are missing one
    // if you are missing one the esp32 will likely crash on invoke
    // best practice it not adding all of them but only the ones you need
    resolver.AddAbs();
    resolver.AddAdd();
    resolver.AddArgMax();
    resolver.AddArgMin();
    resolver.AddAveragePool2D();
    resolver.AddCeil();
    resolver.AddConcatenation();
    resolver.AddConv2D();
    resolver.AddCos();
    resolver.AddDepthwiseConv2D();
    resolver.AddDequantize();
    resolver.AddEqual();
    resolver.AddFloor();
    resolver.AddFullyConnected();
    resolver.AddGreater();
    resolver.AddGreaterEqual();
    resolver.AddHardSwish();
    resolver.AddL2Normalization();
    resolver.AddLess();
    resolver.AddLessEqual();
    resolver.AddLog();
    resolver.AddLogicalAnd();
    resolver.AddLogicalNot();
    resolver.AddLogicalOr();
    resolver.AddLogistic();
    resolver.AddMaximum();
    resolver.AddMaxPool2D();
    resolver.AddMean();
    resolver.AddMinimum();
    resolver.AddMul();
    resolver.AddNeg();
    resolver.AddNotEqual();
    resolver.AddPack();
    resolver.AddPad();
    resolver.AddPadV2();
    resolver.AddPrelu();
    resolver.AddQuantize();
    resolver.AddReduceMax();
    resolver.AddRelu();
    resolver.AddRelu6();
    resolver.AddReshape();
    resolver.AddResizeNearestNeighbor();
    resolver.AddRound();
    resolver.AddRsqrt();
    resolver.AddShape();
    resolver.AddSin();
    resolver.AddSoftmax();
    resolver.AddSplit();
    resolver.AddSplitV();
    resolver.AddSqrt();
    resolver.AddSquare();
    resolver.AddStridedSlice();
    resolver.AddSub();
    resolver.AddSvdf();
    resolver.AddTanh();
    resolver.AddUnpack();

    return TfLiteStatus::kTfLiteOk;
  }
}
/// @brief Writes the jpeg bytes to the serial port as a base64 encoded string
/// use JpegFilter to extract the jpeg bytes and save them to a file
/// @param jpegBytes // the jpeg bytes to write
/// @param jpegSize // the length of the jpeg bytes
void serialWriteJpeg(uint8_t *jpegBytes, size_t jpegSize)
{
  Serial.print("StartJPEG123456");
  size_t outlen;
  size_t base64jpegBufferSize = 30000;
  // Used for base64 encoding jpeg over
  unsigned char *jpegEncodedBuffer = (unsigned char *)ps_malloc(base64jpegBufferSize);
  mbedtls_base64_encode(jpegEncodedBuffer, base64jpegBufferSize, &outlen, jpegBytes, jpegSize);
  Serial.write(jpegEncodedBuffer, outlen);
  Serial.print("EndJPEG123456");
}

/// @brief Updates the jpeg buffer with the current frame
/// from the grayScaleBuffer
void updateJpegBuffer()
{
  ESP_LOGI(TAG, "update jpeg buffer");

  Serial.println("Setup fb");

  free(jpegBytes); // free the previous buffer if any
                   // frame2jpg will malloc the buffer for jpegBytes
                   // gain lock the buffer should not change mid frame
  frame2jpg(grayScalefb, 50, &jpegBytes, &jpegSize);
  // serialWriteJpeg(jpegBytes, jpegSize);
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

int8_t uint8GrayscaleIint8(uint8_t uint8color)
{
  return (int8_t)(((int)uint8color) - 128); // is there a better way?
}
uint8_t int8GrayscaleUint8(int8_t int8color)
{
  return (uint8_t)(((int)int8color) + 128); // is there a better way?
}

uint inferNumberImage(int8_t *mnistimage)
{
  memcpy(input->data.int8, mnistimage, 28 * 28);
  for (int i = 0; i < 28 * 28; i++)
  {
    input->data.int8[i] = mnistimage[i];
  }
  int start = millis();
  error_reporter->Report("Invoking.");

  if (kTfLiteOk != interpreter->Invoke()) // Any error i have in invoke tend to just crash the whole system so i dont usually see this message
  {
    error_reporter->Report("Invoke failed.");
  }
  else
  {
    error_reporter->Report("Invoke passed.");
    error_reporter->Report(" Took :");
    Serial.print(millis() - start);
    error_reporter->Report(" milliseconds");
  }

  TfLiteTensor *output = interpreter->output(0);
  uint result = oneHotDecode(output);
  return result;
}
#ifdef ENABLE_MJPEG
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

/// @brief Draws a hollow rectangle on the frame buffer
void fb_gfx_drawRect(fb_data_t *fb, int32_t x, int32_t y, int32_t w, int32_t h, uint32_t color)
{
  fb_gfx_drawFastHLine(fb, x, y, w, color);
  fb_gfx_drawFastHLine(fb, x, y + h, w, color);
  fb_gfx_drawFastVLine(fb, x, y, h, color);
  fb_gfx_drawFastVLine(fb, x + w, y, h, color);
}
void print_memory_info()
{
  uint32_t free_heap = esp_get_free_heap_size();
  uint32_t total_psram = heap_caps_get_total_size(MALLOC_CAP_SPIRAM);
  uint32_t free_psram = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);

  ESP_LOGI("MemoryInfo", "Free internal heap: %u", free_heap);
  ESP_LOGI("MemoryInfo", "Total PSRAM: %u", total_psram);
  ESP_LOGI("MemoryInfo", "Free PSRAM: %u", free_psram);
}

/// @brief called when a client connects to the mjpeg stream
void handle_jpg_stream(void)
{
  char buf[32];
  int s;

  WiFiClient client = server.client();

  client.write(HEADER, hdrLen);
  client.write(BOUNDARY, bdrLen);

  while (true)
  {
    if (!client.connected())
      break;
    Serial.print("start frame");

    print_memory_info();

    ESP_LOGI(TAG, "Get Camera Frame");

    camera.run();
    ESP_LOGI(TAG, "Got Camera Frame");

    // copy frame
    memcpy(grayScaleBuffer, camera.getfb(), grayScalefb->width * grayScalefb->height);
    ESP_LOGI(TAG, "Copy Camera Frame");

    // convert to uint to int
    // should do this inplace
    ESP_LOGI(TAG, "Conversion to int8");

    int8_t *raw = (int8_t *)ps_malloc(grayScalefb->height * grayScalefb->width * sizeof(int8_t));

    for (int i = 0; i < grayScalefb->width * grayScalefb->height; i++)
    {
      raw[i] = uint8GrayscaleIint8(grayScaleBuffer[i]);
    }
    int8_t *mnist = (int8_t *)ps_malloc(28 * 28 * sizeof(int8_t));

    // preprocess image into minst format
    ESP_LOGI(TAG, "Create Mnist Image style from camera image");

    formatter.CreateMnistImageFromImage(raw, grayScalefb->width, grayScalefb->height, mnist);
    free(raw);
    ESP_LOGI(TAG, "overlay  mnist image");

    for (size_t i = 0; i < 28; i++)
    {
      for (size_t j = 0; j < 28; j++)
      {
        grayScaleBuffer[j + (grayScalefb->width * i)] = int8GrayscaleUint8(mnist[j + (28 * i)]);
      }
    }

    ESP_LOGI(TAG, "infer mnist image");

    uint result = inferNumberImage(mnist);
    ESP_LOGI(TAG, "update image with result %d", result);

    free(mnist);
    fb_data_t fbdata;
    fbdata.data = grayScaleBuffer;
    fbdata.width = grayScalefb->width;
    fbdata.height = grayScalefb->height;
    fbdata.format = FB_GRAY;
    fbdata.bytes_per_pixel = 1;
    char resultString[20];
    sprintf(resultString, "Result: %d", result);
    fb_gfx_print(&fbdata, grayScalefb->width / 4, grayScalefb->height / 4, 127, resultString);
    // fb_gfx_drawRect(&fbdata, grayScalefb->width / 4, grayScalefb->height / 4, 127, 127, 127);
    // delay(10);

    unsigned long startTime = millis();

    updateJpegBuffer();
    unsigned long endTime = millis();
    Serial.print("Time spent updating jpeg buffer:");
    Serial.println(endTime - startTime);

    // delay(100); // TODO set frame rate and try and maintain it
    startTime = millis();
    client.write(CTNTTYPE, cntLen);
    sprintf(buf, "%d\r\n\r\n", jpegSize);
    client.write(buf, strlen(buf));
    client.write((char *)jpegBytes, jpegSize);
    client.write(BOUNDARY, bdrLen);
    endTime = millis();
    unsigned long frameRate = 1000 / (endTime - startTime);
    Serial.print("Frame rate: ");
    Serial.println(frameRate);
    Serial.print("Time spent sending frame");
    Serial.println(endTime - startTime);

    print_memory_info();
    Serial.print("end frame");
  }
}
#endif

#ifdef SOFTAP_MODE
IPAddress apIP = IPAddress(192, 168, 1, 1);
#else
#include "wifikeys.h"
#endif

void initSerial()
{
  Serial.setRxBufferSize(1024);
  Serial.begin(115200);
  Serial.setTimeout(10000);
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
    error_reporter->Report(
        "Model provided is schema version %d not equal "
        "to supported version %d.",
        model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  AllOpsResolver op_resolver;
  RegisterOps(op_resolver);

  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(
      model, op_resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk)
  {
    error_reporter->Report("AllocateTensors() failed");
    return;
  }

  // Get information about the memory area to use for the model's input.
  input = interpreter->input(0);
  error_reporter->Report("Input Shape");
  for (int i = 0; i < input->dims->size; i++)
  {
    error_reporter->Report("%d", input->dims->data[i]);
  }

  error_reporter->Report(TfLiteTypeGetName(input->type));
  error_reporter->Report("Output Shape");

  TfLiteTensor *output = interpreter->output(0);
  for (int i = 0; i < output->dims->size; i++)
  {
    error_reporter->Report("%d", output->dims->data[i]);
  }
  error_reporter->Report(TfLiteTypeGetName(output->type));
  error_reporter->Report("Arena Used:%d bytes of memory", interpreter->arena_used_bytes());
}
void writeGrayScaleBuffer(uint index, uint8_t value)
{
  grayScaleBuffer[index] = value;
}

/// @brief Tests the preloaded images of numbers
void testPreloadedNumbers()
{
  Serial.print("Testing One. Result:");

  uint num = inferNumberImage(number1Sample);
  Serial.print("Testing One. Result:");
  Serial.println(num);
  num = inferNumberImage(number2Sample);
  Serial.print("Testing two. Result:");
  Serial.println(num);
  num = inferNumberImage(number4Sample);
  Serial.print("Testing four. Result:");
  Serial.println(num);
  num = inferNumberImage(number5Sample);
  Serial.print("Testing five. Result:");
  Serial.println(num);
  num = inferNumberImage(number8Sample);
  Serial.print("Testing eight. Result:");
  Serial.println(num);
  num = inferNumberImage(number9Sample);
  Serial.print("Testing nine. Result:");
  Serial.println(num);
}

void updateImageTask(void *param)
{
  while (true)
  {
    delay(1);
  }
}

void setup()
{

  grayScaleBuffer = (uint8_t *)ps_malloc(raw_image_size);

  grayScalefb->buf = grayScaleBuffer;
  grayScalefb->format = PIXFORMAT_GRAYSCALE;
  grayScalefb->len = 0;
  grayScalefb->timestamp = {0, 0};
  grayScalefb->height = 240;
  grayScalefb->width = 320;

  initSerial();
  initTFInterpreter();
  IPAddress ip;
  ESP_LOGI(TAG, "Starting Camera");
  camera.init(esp32cam_ESPCam_config);
  ESP_LOGI(TAG, "Started Camera");

#ifdef SOFTAP_MODE
  const char *hostname = "devcam";
  // WiFi.hostname(hostname); // FIXME - find out why undefined
  ESP_LOGI(TAG, "starting softAP");
  WiFi.mode(WIFI_AP);
  WiFi.softAPConfig(apIP, apIP, IPAddress(255, 255, 255, 0));
  bool result = WiFi.softAP(hostname, "12345678", 1, 0);
  if (!result)
  {
    Serial.println("AP Config failed.");
    return;
  }
  else
  {
    Serial.println("AP Config Success.");
    Serial.print("AP MAC: ");
    Serial.println(WiFi.softAPmacAddress());

    ip = WiFi.softAPIP();
  }
#else
  ESP_LOGI(TAG, "Joining %s", ssid);
  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED)
  {
    delay(500);
    Serial.print(F("."));
  }
  ip = WiFi.localIP();
  Serial.println(F("WiFi connected"));
  Serial.println("");
  Serial.println(ip);
#endif
  const char *ipaddress = ip.toString().c_str();
  ESP_LOGI(TAG, "%s", ipaddress);
  Serial.println("/mjpeg/1");
  server.on("/mjpeg/1", HTTP_GET, handle_jpg_stream);
  server.onNotFound(handleNotFound);
  server.begin();

  for (int i = 0; i < raw_image_size; i++)
  {
    // set every other pixel to black and white
    grayScaleBuffer[i] = (i % 2) ? 0 : 255;
  }
  updateJpegBuffer();
  xTaskCreate(
      updateImageTask,
      "camera_task", 1024 * 2, NULL, 1, NULL);
}

void loop()
{

  server.handleClient();
  delay(10);
}