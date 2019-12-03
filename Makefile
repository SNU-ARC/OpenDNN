include common.mk

CXX_PRJ:=..

SRC:=src
LIB:=lib

all: libopendnn

libopendnn: clean
	@echo USE_CUDA = $(USE_CUDA)
	$(MAKE) -C $(SRC) $@.so; cp $(SRC)/$@.so lib
	@echo "libopendnn.so is built successfully"
	$(MAKE) -C $(SRC) $@.a; cp $(SRC)/$@.a lib
	@echo "libopendnn.a is built successfully"
	@echo "====================================="
	@echo "openDNN is built sucessfully !"
	@echo "Use following commands to set the path"
	@echo "export PATH:=/path/to/include"
	@echo "export LD_LIBRARY_PATH:=/path/to/lib(.so and .a)"

install: libopendnn
	@echo "\n====================================="
	@echo "Warning: This process will copy libopendnn.so into /usr/lib"
	sudo cp lib/* /usr/lib/

clean:
	$(MAKE) -C $(SRC) clean
	$(MUTE) rm -f $(LIB)/*.so $(LIB)/*.a
