project(
  'tflite-app',
  'cpp',
  version : '1.0.0',
  default_options : [
    'cpp_std=c++17',
  ],
)

absl_flags = dependency('absl_flags')
absl_flags_parse = dependency('absl_flags_parse')
absl_strings = dependency('absl_strings')
coral = dependency('coral')
glog = dependency('glog')
edgetpu = dependency('edgetpu')
tensorflow_lite = dependency('tensorflow-lite')
flatbuffers = dependency('flatbuffers')
xtensor = dependency('xtensor')
opencv4 = dependency('opencv4')

dependencies = [
  absl_flags,
  absl_flags_parse,
  absl_strings,
  coral,
  edgetpu,
  glog,
  tensorflow_lite,
  flatbuffers,
  xtensor,
  opencv4
]

executable(
  'tflite-app',
  'main.cpp',
  dependencies : dependencies,
  install : true,
)
