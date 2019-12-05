include common.mk

CXX_PRJ:=..

SRC:=src
LIB:=lib

all: libopendnn

libopendnn: clean
	@mkdir -p lib
	@echo TARGET = $(TARGET)
	$(MAKE) -C $(SRC) $@.so; cp $(SRC)/$@.so lib
	$(MAKE) -C test
	@echo "libopendnn.so is built successfully"
	@echo "====================================="
	@echo "openDNN is built sucessfully !"
	@echo "Use following commands to set the path"
	@echo "export PATH:=/path/to/include"
	@echo "export LD_LIBRARY_PATH:=/path/to/lib(.so)"

install: libopendnn
	@echo "\n====================================="
	@echo "Warning: This process will copy libopendnn.so into /usr/lib"
	sudo cp lib/* /usr/lib/

clean:
	$(MAKE) -C test clean
	$(MAKE) -C $(SRC) clean
	$(MUTE) rm -f $(LIB)/*.so $(LIB)/*.a
