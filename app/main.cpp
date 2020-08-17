#include <iostream>
#include <algorithm>
#include <chrono>  // NOLINT
#include <iostream>
#include <memory>
#include <ostream>
#include <string>

#include "edgetpu.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model.h"

int main(int argc, const char* argv[]) {
  std::unique_ptr<tflite::FlatBufferModel> model =
          tflite::FlatBufferModel::BuildFromFile("./foo.tflite");
  return 0;
}
