#include "edgetpu.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model.h"

int main(int argc, const char* argv[]) {
  auto model = tflite::FlatBufferModel::BuildFromFile("./foo.tflite");
  return 0;
}
