#ifndef MCTS_SEARCH_H
#define MCTS_SEARCH_H

#include <vector>
#include "mcts_node.h"
#include "neural_evaluator.h"
#include "othello/othello.h"

namespace OthelloMCTS {

// MCTS configuration parameters
struct MCTSConfig {
    int num_simulations = 800;
    int batch_size = 32;
    float c_puct = 1.5f;        // Exploration constant
    float virtual_loss = 1.0f;  // For batch diversity
    bool use_dirichlet = true;  // Add noise to root for exploration
    float dirichlet_alpha = 0.3f;
    float dirichlet_epsilon = 0.25f;
};

// Represents one step in the path: the node we were at and the edge we took
struct PathStep {
    Node* node;   // Node we were at
    Edge* edge;   // Edge we took from that node (never null!)
};

// Result of selecting a leaf node
struct SelectResult {
    Node* leaf;                  // The leaf node we reached
    std::vector<PathStep> path;  // Path of decisions (node, edge pairs)
};

class MCTSSearch {
public:
    MCTSSearch(NeuralEvaluator& evaluator, const MCTSConfig& config = MCTSConfig());
    
    // Run search from given position and return best move
    Othello::OthelloMove search(const Othello::OthelloState& root_state);
    
    // Get move visit counts (for analysis or temperature-based selection)
    std::vector<std::pair<Othello::OthelloMove, int>> get_move_visits() const;
    
    // Reset for new game (clears tree)
    void reset();
    
    // Statistics
    int get_nodes_created() const { return pool_.size(); }
    
private:
    NeuralEvaluator& evaluator_;
    MCTSConfig config_;
    NodePool pool_;
    Node* root_ = nullptr;
    
    // Core MCTS phases
    
    // Phase 1: Select leaf node using UCB
    SelectResult select_leaf(Node* root);
    
    // Phase 2: Expand node with NN evaluation
    void expand_node(Node* node, const NNEval& eval);
    
    // Phase 3: Backpropagate value up the tree
    void backpropagate(const SelectResult& result, float value);
    
    // UCB score for edge selection
    float ucb_score(const Edge& edge, int parent_visits) const;
    
    // Select best child using UCB
    Edge* select_best_edge(Node* node) const;
    
    // Lazy child creation when traversing edge
    Node* get_or_create_child(Node* parent, Edge* edge);
    
    // Add Dirichlet noise to root for exploration
    void add_dirichlet_noise(Node* root);
    
    // Convert bitboard to move list
    std::vector<Othello::OthelloMove> bitboard_to_moves(uint64_t bitboard) const;
};

} // namespace OthelloMCTS

#endif // MCTS_SEARCH_H