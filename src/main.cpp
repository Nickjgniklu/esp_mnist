#include <Arduino.h>
#include "main_functions.h"
#include <Wire.h>
#include <SPI.h>
#include <FS.h>
#include <SPIFFS.h>
#include<WiFiClient.h>

#include "image_provider.h"
#include "model_settings.h"
#include "mnist_model.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
//#include "tensorflow/lite/version.h"
#include "camera_config.h"
#include "sampleDigits/sampleDigits.h"

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

TfLiteStatus RegisterOps(AllOpsResolver& resolver) {
  //Not all of these are needed for every model but it can be know you are missing one
  //if you are missing one the esp32 will likely crash on invoke
  //best practice it not adding all of them but only the ones you need
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

void initSerial(){
  Serial.setRxBufferSize(1024);
  Serial.begin(115200);
  Serial.setTimeout(10000);
}

void initTFInterpreter(){
static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;
  //Create Model
  model = tflite::GetModel(mnist_model);
  //Verify Version of Tf Micro matches Model's verson
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
void setup()
{
  initSerial();
  initTFInterpreter();
}

///Returns the index of the max value
uint oneHotDecode(TfLiteTensor *layer){
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

int8_t uint8GrayscaleIint8(uint8_t uint8color){
 return (int8_t)(((int)uint8color) - 128); //is there a better way?
}
uint8_t int8GrayscaleUint8(int8_t int8color){
 return (uint8_t)(((int)int8color) + 128); //is there a better way?
}

uint inferNumberImage(int8_t* mnistimage){
memcpy(input->data.int8, mnistimage, 28 * 28);
  for (int i = 0; i < 28 * 28; i++)
  {
    input->data.int8[i] = mnistimage[i];
  }
  int start = millis();
    error_reporter->Report("Invoking.");

  if (kTfLiteOk != interpreter->Invoke())//Any error i have in invoke tend to just crash the whole system so i dont usually see this message
  {
    error_reporter->Report("Invoke failed.");
  }
  else
  {
    error_reporter->Report("Invoke passed.");
    error_reporter->Report(" Took :");
    Serial.print(millis()-start);
    error_reporter->Report(" milliseconds");

  }
  
  TfLiteTensor *output = interpreter->output(0);
  uint result= oneHotDecode(output);
  return result;
}

void testPreloadedNumbers(){
  Serial.print("Testing One. Result:");

  uint num=inferNumberImage(number1Sample);
  Serial.print("Testing One. Result:");
  Serial.println(num);
  num=inferNumberImage(number2Sample);
  Serial.print("Testing two. Result:");
  Serial.println(num);
  num=inferNumberImage(number4Sample);
  Serial.print("Testing four. Result:");
  Serial.println(num);
  num=inferNumberImage(number5Sample);
  Serial.print("Testing five. Result:");
  Serial.println(num);
  num=inferNumberImage(number8Sample);
  Serial.print("Testing eight. Result:");
  Serial.println(num);
  num=inferNumberImage(number9Sample);
  Serial.print("Testing nine. Result:");
  Serial.println(num);
}
void loop()
{
  delay(10000);

  testPreloadedNumbers();
}