CXX := clang++

RM := rm -f
CFLAGS := -I. $(shell pkg-config --cflags opencv4)
OPT := 3
DEBUG :=
LDFLAGS := -flto
CXXFLAGS := -O$(OPT) -std=c++17 $(DEBUG) -flto

POSE_LDLIBS := $(shell pkg-config --libs-only-l opencv4) \
    -lboost_program_options \
    -ledgetpu \
    -ledgetpu_basic_engine \
    -ledgetpu_basic_engine_native \
    -ltensorflow-lite \
    -lrt \
    -lpthread \
    -ldl

MODEL_INFO_LDLIBS := -lboost_program_options \
    -lflatbuffers \
    -lfmt \
    -lrt \
    -lpthread \
    -ldl

POSE_SRCS := $(wildcard pose_src/*.cc)
POSE_OBJS := $(subst .cc,.o,$(POSE_SRCS))

MODEL_INFO_SRCS := $(wildcard model-info_src/*.cc)
MODEL_INFO_OBJS := $(subst .cc,.o,$(MODEL_INFO_SRCS))

all: pose model-info

%.o: %.cc
	$(CXX) $(CFLAGS) $(CXXFLAGS) -c $^ -o $@

pose: $(POSE_OBJS)
	$(CXX) -o $@ $^ $(LDFLAGS) $(POSE_LDLIBS)

model-info: $(MODEL_INFO_OBJS)
	$(CXX) -o $@ $^ $(LDFLAGS) $(MODEL_INFO_LDLIBS)

OBJS := $(POSE_OBJS) $(MODEL_INFO_OBJS)

clean:
	$(RM) $(OBJS)

distclean: clean
	$(RM) *~
