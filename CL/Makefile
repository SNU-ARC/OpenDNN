CXX:=gcc
CFLAGS:=-O3
LDFLAGS:=-lOpenCL -lm
LIB:=libopendnn.so
current_dir=$(shell pwd)

BIN:=test.exe
OBJ:=opendnn.o

all: $(BIN) $(LIB)

main.o: main.cpp
	$(CXX) -c $< -o $@ $(CFLAGS)

$(OBJ): opendnn.c
	$(CXX) -Wall -fPIC -c $< -o $@ $(CFLAGS) $(LDFLAGS)

$(BIN): main.o $(OBJ)
	$(CXX) $^ -o $@ $(CFLAGS) $(LDFLAGS) -lopendnn

$(LIB): $(OBJ)
	$(CXX) -shared -o $@ $<

.PHONY: clean

install:
	sudo mv $(LIB) /usr/lib/$(LIB)
	sudo cp opendnn_kernel.cl /usr/lib/opendnn_kernel.cl
	sudo cp opendnn.h /usr/include/opendnn.h

clean:
	rm -f *.o *.so test.exe
