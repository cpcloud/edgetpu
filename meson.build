project('tflite-app', 'c++')

srcs = ['app/main.cpp']

boost_program_options = dependency('boost')
edgetpu = dependency('edgetpu')
edgetpu_basic_engine = dependency('edgetpu_basic_engine')
edgetpu_basic_engine_native = dependency('edgetpu_basic_engine_native')
tensorflow_lite = dependency('tensorflow_lite')
flatbuffers = dependency('flatbuffers')
xtensor = dependency('xtensor')
opencv4 = dependency('opencv4')

dependencies = [
  boost_program_options,
  edgetpu,
  edgetpu_basic_engine,
  edgetpu_basic_engine_native,
  tensorflow_lite,
  flatbuffers,
  xtensor,
  opencv4
]

executable('tflite-app', srcs, dependencies : dependencies)
