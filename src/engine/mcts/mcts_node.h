#ifndef MCTS_NODE_H
#define MCTS_NODE_H

#include <vector>
#include <memory>
#include "othello/othello.h"

namespace OthelloMCTS {

struct Edge;

struct Node {
    Othello::OthelloState state;
    
    // Tree structure
    std::vector<Edge> edges;
    // NO parent pointer - we store path during traversal
    
    // Statistics (single-threaded, no atomics needed!)
    int visit_count = 0;
    float total_value = 0.0f;
    
    // Expansion status
    bool expanded = false;
    
    // Helper methods
    bool is_terminal() const { return Othello::OthelloOps::isTerminal(state); }
    float Q() const { return visit_count > 0 ? total_value / visit_count : 0.0f; }
};

struct Edge {
    Othello::OthelloMove move;
    float prior = 0.0f;  // Policy probability from NN
    Node* child = nullptr;  // Lazy creation - null until traversed
    
    // Statistics
    int visit_count = 0;
    float total_value = 0.0f;
    
    // UCB score calculation
    float Q() const { return visit_count > 0 ? total_value / visit_count : 0.0f; }
};

// Memory pool for efficient node allocation
class NodePool {
public:
    NodePool() { nodes_.reserve(100000); }  // Pre-reserve space
    
    Node* allocate(const Othello::OthelloState& state);
    void reset();
    size_t size() const { return nodes_.size(); }
    
private:
    std::vector<std::unique_ptr<Node>> nodes_;
};

} // namespace OthelloMCTS

#endif // MCTS_NODE_H