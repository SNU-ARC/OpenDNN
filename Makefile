include common.mk

CXX_PRJ:=..

SRC:=src
LIB:=lib

all: libopendnn

libopendnn: clean
	@mkdir -p lib
	@echo "TARGET: $(TARGET)"
	$(MAKE) -C $(SRC) $@.so
	$(MAKE) -C test
	@echo "libopendnn.so is built successfully"
	@echo "=====================================\\n"
	@echo "Use following commands to set the path or"
	@echo "run make install for copying to system path\\n"
	@echo "export PATH:=/path/to/include/opendnn.h"
	@echo "export LD_LIBRARY_PATH:=/path/to/libopendnn.so"

install: libopendnn
	@echo "\n====================================="
	@echo "Warning: This process will copy libopendnn.so and opendnn.h into system path"
	sudo cp lib/* /usr/lib/
	sudo cp include/* /usr/include/

clean:
	$(MAKE) -C test clean
	$(MAKE) -C $(SRC) clean
	$(MUTE) rm -f $(LIB)/*.so $(LIB)/*.a
