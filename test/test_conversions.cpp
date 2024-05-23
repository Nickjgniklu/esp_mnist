#include "unity.h"
#include "Arduino.h"
#include <WiFiClient.h>
#include <ConversionTools.h>
#include <ImageFormater.h>
#include "esp_camera.h"

#include "example_image.h"
ImageFormater formatter;

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

// TODO FIX THIS TEST EXAMPLE IS NOT A VALID IMAGE
void test_format_mnist_image(void)
{
    size_t height = 240;
    size_t width = 320;
    camera_fb_t *grayScalefb = new camera_fb_t();
    grayScalefb->buf = __output_jpg;
    grayScalefb->format = PIXFORMAT_JPEG;
    grayScalefb->len = __output_jpg_len;
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

    // validate 28x28 image is only black and white
    for (size_t i = 0; i < 28 * 28; i++)
    {
        TEST_ASSERT_TRUE_MESSAGE(mnist[i] == -128 || mnist[i] == 127, "mnist image is not black and white");
    }
    // validate 28x28 image has 4 black border around the 20x20 center
    // for (size_t i = 0; i < 28; i++)
    // {
    //     for (size_t j = 0; j < 28; j++)
    //     {
    //         if (i < 4 || i > 23 || j < 4 || j > 23)
    //         {
    //             TEST_ASSERT_TRUE_MESSAGE(mnist[i * 28 + j] == -128, "mnist image has bad border");
    //         }
    //     }
    // }
    free(mnist);
}

int runUnityTests(void)
{
    UNITY_BEGIN();
    RUN_TEST(test_convert_uint8_to_int8);
    RUN_TEST(test_convert_int8_to_uint8);
    // RUN_TEST(test_format_mnist_image);
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
