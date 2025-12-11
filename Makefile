CXX = g++
CXXFLAGS = -std=c++17 -Iexternal -Isrc

SRC_DIR = src
OBJ_DIR = build
BIN_DIR = bin
TEST_DIR = tests
TEST_OBJ_DIR = build_tests

SOURCES = $(shell find $(SRC_DIR) -type f -name '*.cpp')
SOURCES_NO_MAIN = $(filter-out $(SRC_DIR)/main.cpp, $(SOURCES))
OBJECTS = $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(SOURCES))
EXECUTABLE = $(BIN_DIR)/engine

TEST_SOURCES = $(shell find $(TEST_DIR) -type f -name '*.cpp')
TEST_OBJECTS = $(patsubst $(TEST_DIR)/%.cpp,$(TEST_OBJ_DIR)/%.o,$(TEST_SOURCES))
TEST_LINK_OBJECTS = $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(SOURCES_NO_MAIN))
TEST_EXECUTABLE = $(BIN_DIR)/tests


all: $(EXECUTABLE)
	#@echo $(OBJECTS)

$(EXECUTABLE): $(OBJECTS)
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $^

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(dir $@)
	#@echo $^
	$(CXX) $(CXXFLAGS) -c -o $@ $^

tests: $(TEST_EXECUTABLE)
	./$(TEST_EXECUTABLE)

$(TEST_EXECUTABLE): $(TEST_OBJECTS) $(TEST_LINK_OBJECTS)
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $^

$(TEST_OBJ_DIR)/%.o: $(TEST_DIR)/%.cpp
	@mkdir -p $(dir $@)
	#@echo $^
	$(CXX) $(CXXFLAGS) -c -o $@ $^


clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR) $(TEST_OBJ_DIR)

run: $(EXECUTABLE)
	@./$(EXECUTABLE)

.PHONY:
	clean all