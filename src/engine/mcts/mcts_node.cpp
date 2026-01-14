#include "mcts_node.h"

namespace OthelloMCTS {

bool Node::is_terminal() const {
    return Othello::OthelloOps::isTerminal(state);
}

Node* NodePool::allocate(const Othello::OthelloState& state) {
    nodes_.emplace_back(std::make_unique<Node>());
    nodes_.back()->state = state;
    return nodes_.back().get();
}

void NodePool::reset() {
    nodes_.clear();
}

} // namespace OthelloMCTS