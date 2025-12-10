#include <HTTPClient.h>

const char *uploadServer = "http://172.20.10.10:5000/upload"; // Replace with your API

void sendFrameToServer() {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("WiFi not connected");
    return;
  }

  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("Camera capture failed");
    return;
  }

  Serial.printf("Captured %u bytes\n", fb->len);

  HTTPClient http;
  http.begin(uploadServer);
  http.addHeader("Content-Type", "image/jpeg");

  int code = http.POST(fb->buf, fb->len);
  if (code > 0) {
    Serial.printf("Image uploaded! Server response: %d\n", code);
  } else {
    Serial.printf("Upload failed: %s\n", http.errorToString(code).c_str());
  }

  http.end();
  esp_camera_fb_return(fb);
}
