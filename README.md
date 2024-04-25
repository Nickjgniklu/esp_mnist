# a simple demo of ESP_TF
# Supported boards
The default board for this example is the
`Seeed XIAO ESP32S3 Sense`

To change the board edit
platformio.ini

An config for the esp32cam (non ESP32S3) is provided in platformio.ini

Most ESP32 boards should work do not pass S3 build flags to non S3 boards.

## Training your own model
See ESP_TF_demo.ipynb

## performance 
The included model takes :

~1390 milliseconds to invoke on a ESP32_S3 without ESP_NN (tensorflow micro example implementations of layers)

~217 milliseconds to invoke on a ESP32_S3 with ESP_NN
With espS3 optimizations (espressif ansi c implementations)

~21 milliseconds to invoke on a ESP32_S3 with ESP_NN (espressif s3 specific implementations)
