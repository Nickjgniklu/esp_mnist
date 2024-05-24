#include "Arduino.h"
#include <WiFiClient.h>
#include <ConversionTools.h>
#include <ImageFormater.h>
#include "esp_camera.h"
#include <mnist_model.h>
#include <all_ops_resolver.h>

#include "tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "example_color.h"
#include "example_mnist.h"
#include "unity.h"

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

void initTFInterpreter()
{
    // TODO assert with unity that this init worked well
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
    CREATE_ALL_OPS_RESOLVER(op_resolver)
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
void setUp(void)
{
    // set stuff up here
}

void tearDown(void)
{
    // clean stuff up here
}

void test_convert_uint8_to_int8(void)
{
    uint8_t input[5] = {0, 1, 2, 3, 255};
    int8_t expected_output[5] = {-128, -127, -126, -125, 127};
    int8_t out[5];
    ConversionTools::uint8_to_int8(input, out, 5);
    TEST_ASSERT_EQUAL_INT8_ARRAY(expected_output, out, 5);
}

void test_convert_int8_to_uint8(void)
{
    int8_t input[5] = {-128, -127, -126, -125, 127};
    uint8_t expected_output[5] = {0, 1, 2, 3, 255};
    uint8_t out[5];
    ConversionTools::int8_to_uint8(input, out, 5);
    TEST_ASSERT_EQUAL_UINT8_ARRAY(expected_output, out, 5);
}

void test_format_mnist_image(void)
{
    size_t height = 240;
    size_t width = 320;
    camera_fb_t *grayScalefb = new camera_fb_t();
    grayScalefb->buf = __example_color_jpeg;
    grayScalefb->format = PIXFORMAT_JPEG;
    grayScalefb->len = __example_color_jpeg_len;
    grayScalefb->timestamp = {0, 0};
    grayScalefb->height = height;
    grayScalefb->width = width;
    uint8_t *color_raw = (uint8_t *)ps_malloc(width * height * 3 * sizeof(uint8_t));
    size_t color_raw_size = 0;

    frame2bmp(grayScalefb, &color_raw, &color_raw_size);

    // Convert to grayscale
    uint8_t *gray_raw = (uint8_t *)ps_malloc(width * height * sizeof(uint8_t));
    for (size_t i = 0; i < width * height; i++)
    {
        gray_raw[i] = (color_raw[i * 3] + color_raw[i * 3 + 1] + color_raw[i * 3 + 2]) / 3;
    }
    free(color_raw);

    int8_t *int_raw = (int8_t *)ps_malloc(width * height * sizeof(int8_t));
    ConversionTools::uint8_to_int8(gray_raw, int_raw, width * height);
    free(gray_raw);

    int8_t *mnist = (int8_t *)ps_malloc(28 * 28 * sizeof(int8_t));

    formatter.CreateMnistImageFromImage(int_raw, width, height, mnist);
    free(int_raw);

    // validate 28x28 image has 4 black border around the 20x20 center
    for (size_t i = 0; i < 28; i++)
    {
        for (size_t j = 0; j < 28; j++)
        {
            if (i < 4 || i > 23 || j < 4 || j > 23)
            {
                TEST_ASSERT_TRUE_MESSAGE(mnist[i * 28 + j] == -128, "mnist image has bad border");
            }
        }
    }

    free(mnist);
}
void test_tensorflow_mnist_conv(void)
{
    initTFInterpreter();
    memcpy(input->data.int8, __example_mnist, __example_mnist_length);

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
    TEST_ASSERT_EQUAL_UINT(3, result);
}

int runUnityTests(void)
{
    UNITY_BEGIN();
    RUN_TEST(test_convert_uint8_to_int8);
    RUN_TEST(test_convert_int8_to_uint8);
    RUN_TEST(test_format_mnist_image);
    RUN_TEST(test_tensorflow_mnist_conv);
    return UNITY_END();
}

void setup()
{
    // Wait ~2 seconds before the Unity test runner
    // establishes connection with a board Serial interface
    delay(2000);

    runUnityTests();
}
void loop() {}
