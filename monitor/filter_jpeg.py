import base64
import os
from platformio.public import DeviceMonitorFilterBase
import datetime


class JpegFilter(DeviceMonitorFilterBase):
    NAME = "JpegFilter"


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.buffer = ""
        self.image_count = 0
        if not os.path.isdir("logged_jpegs"):
            os.makedirs("logged_jpegs")



    def rx(self, text):
        self.buffer += text
        start_marker = "StartJPEG123456"
        end_marker = "EndJPEG123456"
        start_index = self.buffer.find(start_marker)
        end_index = self.buffer.find(end_marker)

        if start_index != -1 and end_index != -1:
            jpeg_data = self.buffer[start_index + len(start_marker):end_index]
            log_file_name = os.path.join(
                "logged_jpegs", f"output{self.image_count}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.jpeg"
            )
            self.image_count = self.image_count + 1
            with open(log_file_name, "wb") as f:
                f.write(base64.b64decode(jpeg_data))
            self.buffer = self.buffer[end_index + len(end_marker):]
            return f"Saved: {log_file_name}\n" + self.buffer
        if start_index != -1:
            ""
        return text

    def tx(self, text):
        #print(f"Sent: {text}\n")
        return text