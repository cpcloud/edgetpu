CXX = clang++
SRCS = $(wildcard *.cc)
RM = rm -f
CFLAGS = -I. $(shell pkg-config --cflags opencv4)
OPT = 3
# DEBUG = -ggdb
CXXFLAGS = -O$(OPT) -std=c++17 $(DEBUG)
LDFLAGS = $(shell pkg-config --libs-only-L opencv4)
LDLIBS = $(shell pkg-config --libs-only-l opencv4) \
    -lboost_program_options \
    -ledgetpu \
    -ledgetpu_basic_engine \
    -ledgetpu_basic_engine_native \
    -ltensorflow-lite \
    -lrt \
    -lpthread \
    -ldl
OBJS = $(subst .cc,.o,$(SRCS))

all: tflite-app

%.o: %.cc
	$(CXX) $(CFLAGS) $(CXXFLAGS) -c $^ -o $@

tflite-app: $(OBJS)
	$(CXX) $(LDFLAGS) -o $@ $(OBJS) $(LDLIBS)

clean:
	$(RM) $(OBJS)

distclean: clean
	$(RM) *~
