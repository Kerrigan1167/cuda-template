# OSC CUDA Makefile Template

NVCC          = nvcc
CUDA_FLAGS    = -Wno-deprecated-gpu-targets
COMPILE       = $(NVCC) $(CUDA_FLAGS)

CUDA_SOURCES  = cudatemplate.cu
OBJECTS       = $(CUDA_SOURCES:.cu=.o)
TARGET        = cudatemplate

all: $(TARGET)

$(TARGET): $(OBJECTS)
	@$(COMPILE) $^ -o $@

%.o:%.cu
	@$(COMPILE) -c $< -o $@

clean:
	@rm -rf *.o $(TARGET)

