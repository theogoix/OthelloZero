#include <iostream>
#include "game/othello/othello.h"

int main() {
    std::cout << SECRET_NUMBER;
    hello_world();
    OthelloState state = OthelloOps::initialState();
    print_othello_state(state);
    std::cout << state.currentDiscs;
    std::cout << "hello world\n";
}