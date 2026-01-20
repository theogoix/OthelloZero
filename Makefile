CXX = g++
PYTHON = python.3.12
PYTHON_CONFIG = python3.12-config

OPT ?= 0
CXXFLAGS = -std=c++17 -Wall -Wextra -g -O$(OPT) -Iexternal -Isrc -fPIC -Wno-unused-parameter

VENV := .venv
PY   := $(VENV)/bin/python
PIP  := $(VENV)/bin/pip
PYBIND11_INCLUDES = $(shell $(PY) -m pybind11 --includes)

PYTHON_SUFFIX := $(shell $(PYTHON_CONFIG) --extension-suffix)
SRC_DIR = src
BUILD_DIR := build


LIBTORCH_INCLUDE := -Iexternal/libtorch/include \
	-Iexternal/libtorch/include/torch/csrc/api/include

LIBTORCH_LIB_DIR := -Lexternal/libtorch/lib
LIBTORCH_LIBS := \
  -Wl,--no-as-needed \
  -ltorch -ltorch_cuda -lc10_cuda -ltorch_cpu -lc10 \
  -Wl,--as-needed \
  -Wl,-rpath,external/libtorch/lib

MAIN_DIR = $(SRC_DIR)/mains

OTHELLO_SRC_DIR = $(SRC_DIR)/othello
OTHELLO_SOURCES = $(shell find $(OTHELLO_SRC_DIR) -type f -name '*.cpp')
OTHELLO_BUILD_DIR = $(BUILD_DIR)/othello
OTHELLO_OBJ = $(patsubst $(OTHELLO_SRC_DIR)/%.cpp,$(OTHELLO_BUILD_DIR)/%.o,$(OTHELLO_SOURCES))

GUI_DIR := src/gui

BINDINGS_SRC_DIR = $(SRC_DIR)/bindings
BINDINGS_SOURCES = $(shell find $(BINDINGS_SRC_DIR) -type f -name '*.cpp')
BINDINGS_SHARED_DIR = $(GUI_DIR)
BINDINGS_BUILD_DIR = $(BUILD_DIR)/bindings
BINDINGS_OBJ = $(patsubst $(BINDINGS_SRC_DIR)/%.cpp,$(BINDINGS_BUILD_DIR)/%.o,$(BINDINGS_SOURCES))
BINDINGS_SHARED_OBJ = $(patsubst $(BINDINGS_SRC_DIR)/%.cpp,$(BINDINGS_SHARED_DIR)/%$(PYTHON_SUFFIX),$(BINDINGS_SOURCES))
BINDINGS_LINK_OBJECTS = $(OTHELLO_OBJ)
BINDINGS_BUILD_FLAGS = $(PYBIND11_INCLUDES)
BINDINGS_SHARED_FLAGS = $(shell $(PYTHON_CONFIG) --ldflags --embed)

ENGINE_SRC_DIR = $(SRC_DIR)/engine
ENGINE_SOURCES = $(shell find $(ENGINE_SRC_DIR) -type f -name '*.cpp')
ENGINE_BUILD_DIR = $(BUILD_DIR)/engine
ENGINE_OBJ = $(patsubst $(ENGINE_SRC_DIR)/%.cpp,$(ENGINE_BUILD_DIR)/%.o,$(ENGINE_SOURCES))

TRAINING_SRC_DIR = $(SRC_DIR)/training
TRAINING_SOURCES = $(shell find $(TRAINING_SRC_DIR) -type f -name '*.cpp')
TRAINING_BUILD_DIR = $(BUILD_DIR)/training
TRAINING_OBJ = $(patsubst $(TRAINING_SRC_DIR)/%.cpp,$(TRAINING_BUILD_DIR)/%.o,$(TRAINING_SOURCES))



TEST_SRC_DIR := tests
TEST_BUILD_DIR := build_tests
TEST_SOURCES = $(shell find $(TEST_SRC_DIR) -type f -name '*.cpp')
TEST_OBJECTS = $(patsubst $(TEST_SRC_DIR)/%.cpp,$(TEST_OBJ_DIR)/%.o,$(TEST_SOURCES))
TEST_LINK_OBJECTS = $(OTHELLO_OBJ)
TEST_EXECUTABLE = $(BIN_DIR)/tests

OBJ_DIR := build
BIN_DIR := bin
SHARE_DIR := $(OBJ_DIR)/pyext


EXECUTABLE = $(BIN_DIR)/engine
EXECUTABLE_MAIN = $(MAIN_DIR)/main.cpp
DATAGEN_EXE = $(BIN_DIR)/datagen
DATAGEN_MAIN = $(MAIN_DIR)/generate_selfplay.cpp

TEST_SOURCES = $(shell find $(TEST_DIR) -type f -name '*.cpp')
TEST_OBJECTS = $(patsubst $(TEST_DIR)/%.cpp,$(TEST_OBJ_DIR)/%.o,$(TEST_SOURCES))
TEST_LINK_OBJECTS = $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(SOURCES_NO_MAIN_NO_BINDINGS))
TEST_EXECUTABLE = $(BIN_DIR)/tests






all: $(EXECUTABLE)
	#@echo $(OTHELLO_OBJ)

training: $(DATAGEN_EXE)

$(VENV):
	python -m venv $(VENV)

install: $(VENV)
	$(PIP) install -r requirements.txt

othello: $(OTHELLO_OBJ)

$(OTHELLO_BUILD_DIR)/%.o: $(OTHELLO_SRC_DIR)/%.cpp
	@mkdir -p $(dir $@)
	#@echo $^
	$(CXX) $(CXXFLAGS) -c -o $@ $^

$(ENGINE_BUILD_DIR)/%.o: $(ENGINE_SRC_DIR)/%.cpp
	@mkdir -p $(dir $@)
	#@echo $^
	$(CXX) $(CXXFLAGS) $(LIBTORCH_INCLUDE) -c -o $@ $^

$(TRAINING_BUILD_DIR)/%.o: $(TRAINING_SRC_DIR)/%.cpp
	@mkdir -p $(dir $@)
	#@echo $^
	$(CXX) $(CXXFLAGS) $(LIBTORCH_INCLUDE) -c -o $@ $^


show:
	echo $(PYTHON_SUFFIX)
	echo $(BINDINGS_SHARED_FLAGS)

gui: bindings
	$(PY) $(GUI_DIR)/gui.py

bindings: $(BINDINGS_SHARED_OBJ)



$(BINDINGS_SHARED_DIR)/%$(PYTHON_SUFFIX): $(BINDINGS_BUILD_DIR)/%.o $(BINDINGS_LINK_OBJECTS)
	@mkdir -p $(dir $@)
	#@echo $^
	$(CXX) $(CXXFLAGS) -shared -o $@ $^ $(BINDINGS_SHARED_FLAGS)

$(BINDINGS_BUILD_DIR)/%.o: $(BINDINGS_SRC_DIR)/%.cpp
	@mkdir -p $(dir $@)
	#@echo $^
	$(CXX) $(CXXFLAGS) $(PYBIND11_INCLUDES) -fPIC -c $^ -o $@ $(BINDINGS_BUILD_FLAGS)




$(EXECUTABLE): $(OTHELLO_OBJ) $(ENGINE_OBJ)
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $(LIBTORCH_INCLUDE) -o $@ $^ $(EXECUTABLE_MAIN) $(LIBTORCH_LIB_DIR) $(LIBTORCH_LIBS)

$(DATAGEN_EXE): $(OTHELLO_OBJ) $(ENGINE_OBJ) $(TRAINING_OBJ)
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $(LIBTORCH_INCLUDE) -o $@ $^ $(DATAGEN_MAIN) $(LIBTORCH_LIB_DIR) $(LIBTORCH_LIBS)



tests: $(TEST_EXECUTABLE)
	./$(TEST_EXECUTABLE)

$(TEST_EXECUTABLE): $(TEST_OBJECTS) $(TEST_LINK_OBJECTS)
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $^

$(TEST_OBJ_DIR)/%.o: $(TEST_DIR)/%.cpp
	@mkdir -p $(dir $@)
	#@echo $^
	$(CXX) $(CXXFLAGS) -c -o $@ $^


debug: $(EXECUTABLE)


clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR) $(TEST_OBJ_DIR)
	rm -f $(BINDINGS_SHARED_DIR)/*.so

clean_executable:
	rm -rf $(BIN_DIR)

run: $(EXECUTABLE)
	@./$(EXECUTABLE)

.PHONY:
	clean all run clean_executable