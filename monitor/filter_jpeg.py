import base64
import os
from platformio.public import DeviceMonitorFilterBase
import datetime


class JpegFilter(DeviceMonitorFilterBase):
    NAME = "JpegFilter"


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.buffer = ""
        self.reading_jpeg = False
        self.image_count = 0
        if not os.path.isdir("logged_jpegs"):
            os.makedirs("logged_jpegs")

    def set_running_terminal(self, terminal):
        # force to Latin-1, issue #4732
        # without this encoding some bytes are lost
        if terminal.input_encoding == "UTF-8":
            terminal.set_rx_encoding("Latin-1")
        super().set_running_terminal(terminal)

    def rx(self, text):
        self.buffer += text
        start_marker = "StartJPEG123456"
        end_marker = "EndJPEG123456"
        start_index = self.buffer.find(start_marker)
        output = ""
        if start_index != -1:
            output = self.buffer[:start_index]
            self.reading_jpeg = True
        end_index = self.buffer.find(end_marker)

        if start_index != -1 and end_index != -1:
            jpeg_data = self.buffer[start_index + len(start_marker):end_index]
            log_file_name = os.path.join(
                "logged_jpegs", f"output{self.image_count}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.jpeg"
            )
            self.image_count = self.image_count + 1
            with open(log_file_name, "wb") as f:
                f.write(jpeg_data.encode("latin1"))
            self.buffer = self.buffer[end_index + len(end_marker):]
            self.reading_jpeg = False
            return f"Saved: {log_file_name}\n" + self.buffer

        if self.reading_jpeg:
            return output
        else:
            return text

    def tx(self, text):
        #print(f"Sent: {text}\n")
        return text