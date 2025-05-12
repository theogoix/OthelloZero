CXX = g++
CXXFLAGS = -std=c++17 

SRC_DIR = src
OBJ_DIR = build
BIN_DIR = bin

SOURCES = $(shell find $(SRC_DIR) -type f -name '*.cpp')
OBJECTS = $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(SOURCES))
EXECUTABLE = $(BIN_DIR)/engine




all: $(EXECUTABLE)
	echo $(OBJECTS)

$(EXECUTABLE): $(OBJECTS)
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $^

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(dir $@)
	echo $^
	$(CXX) $(CXXFLAGS) -c -o $@ $^

clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)

run: $(EXECUTABLE)
	@./$(EXECUTABLE)

.PHONY:
	clean all