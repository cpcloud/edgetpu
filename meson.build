project(
  'tflite-app',
  'cpp',
  version : '1.0.0',
  default_options : [
    'cpp_std=c++11',
    'b_lto=true',
  ],
)

boost_program_options = dependency('boost')
edgetpu = dependency('edgetpu')
coral = dependency('coral')
tensorflow_lite = dependency('tensorflow-lite')
flatbuffers = dependency('flatbuffers')
xtensor = dependency('xtensor')
opencv4 = dependency('opencv4')

dependencies = [
  boost_program_options,
  edgetpu,
  tensorflow_lite,
  flatbuffers,
  xtensor,
  opencv4
]

executable(
  'tflite-app',
  'main.cpp',
  dependencies : dependencies,
)
