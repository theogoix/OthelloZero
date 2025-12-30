CXX = g++
PYTHON = python.3.12
PYTHON_CONFIG = python3.12-config
CXXFLAGS = -std=c++17 -Wall -Wextra -g -O0 -Iexternal -Isrc

VENV := .venv
PY   := $(VENV)/bin/python
PIP  := $(VENV)/bin/pip
PYBIND11_INCLUDES = $(shell $(PY) -m pybind11 --includes)

PYTHON_SUFFIX = $(shell $(PY)-config --extension-suffix)
SRC_DIR = src
BUILD_DIR := build


OTHELLO_SRC_DIR = $(SRC_DIR)/othello
OTHELLO_SOURCES = $(shell find $(OTHELLO_SRC_DIR) -type f -name '*.cpp')
OTHELLO_BUILD_DIR = $(BUILD_DIR)/othello
OTHELLO_OBJ = $(patsubst $(OTHELLO_SRC_DIR)/%.cpp,$(OTHELLO_BUILD_DIR)/%.o,$(OTHELLO_SOURCES))


BINDINGS_SRC_DIR = $(SRC_DIR)/bindings
BINDINGS_SOURCES = $(shell find $(BINDINGS_SRC_DIR) -type f -name '*.cpp')
BINDINGS_SHARED_DIR = src/gui
BINDINGS_BUILD_DIR = $(BUILD_DIR)/bindings
BINDINGS_OBJ = $(patsubst $(BINDINGS_SRC_DIR)/%.cpp,$(BINDINGS_BUILD_DIR)/%.o,$(BINDINGS_SOURCES))
BINDINGS_SHARED_OBJ = $(patsubst $(BINDINGS_SRC_DIR)/%.cpp,$(BINDINGS_SHARED_DIR)/%.so,$(BINDINGS_SOURCES))
BINDINGS_LINK_OBJECTS = $(OTHELLO_OBJ)
BINDINGS_BUILD_FLAGS = $(PYBIND11_INCLUDES) -fPIC
BINDINGS_SHARED_FLAGS = -shared $(shell $(PYTHON_CONFIG) --ldflags)

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

TEST_SOURCES = $(shell find $(TEST_DIR) -type f -name '*.cpp')
TEST_OBJECTS = $(patsubst $(TEST_DIR)/%.cpp,$(TEST_OBJ_DIR)/%.o,$(TEST_SOURCES))
TEST_LINK_OBJECTS = $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(SOURCES_NO_MAIN_NO_BINDINGS))
TEST_EXECUTABLE = $(BIN_DIR)/tests






all: $(EXECUTABLE)
	#@echo $(OTHELLO_OBJ)


$(VENV):
	python -m venv $(VENV)

install: $(VENV)
	$(PIP) install -r requirements.txt

othello: $(OTHELLO_OBJ)

$(OTHELLO_BUILD_DIR)/%.o: $(OTHELLO_SRC_DIR)/%.cpp
	@mkdir -p $(dir $@)
	#@echo $^
	$(CXX) $(CXXFLAGS) -c -o $@ $^


bindings: $(BINDINGS_SHARED_OBJ)


$(BINDINGS_SHARED_DIR)/%.so: $(BINDINGS_BUILD_DIR)/%.o
	@mkdir -p $(dir $@)
	#@echo $^
	$(CXX) $(CXXFLAGS) $(BINDINGS_SHARED_FLAGS) -o $@ $^

$(BINDINGS_BUILD_DIR)/%.o: $(BINDINGS_SRC_DIR)/%.cpp $(BINDINGS_LINK_OBJECTS)
	@mkdir -p $(dir $@)
	#@echo $^
	$(CXX) $(CXXFLAGS) -o $@ $^ $(BINDINGS_BUILD_FLAGS)




$(EXECUTABLE): $(OTHELLO_OBJ)
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(SRC_DIR)/main.cpp


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

run: $(EXECUTABLE)
	@./$(EXECUTABLE)

.PHONY:
	clean all