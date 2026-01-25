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


struct MCTSSearchResult {
    Othello::OthelloMove selected_move;
    Othello::OthelloState state;
    int total_visits;
    std::array<int, 64> visit_counts = {};
    std::array<float, 64> q_values = {};

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
    
    // Move selection with temperature
    Othello::OthelloMove select_move(Node* root, float temperature) const;
    
    // Run search from given position and return best move
    Node* search(const Othello::OthelloState& root_state, Node* root = nullptr);

    // Run search from given position and return best move
    Node* search_async(const Othello::OthelloState& root_state, Node* root = nullptr);
    
    // Extract information from root (after search)
    std::array<int, 64> get_visit_counts(Node* root) const;
    std::array<float, 64> get_policy(Node* root) const;
    std::array<float, 64> get_q_values(Node* root) const;
    float get_q_value_root(Node* root) const;


    // Tree reuse: get child node for move (returns nullptr if not found)
    Node* get_child_node(Node* root, Othello::OthelloMove move);


    // Get statistics
    int get_total_visits(Node* root) const { return root->visit_count; }
    
    // Reset for new game (clears tree)
    void reset();
    
    // Statistics
    int get_nodes_created() const { return pool_.size(); }
    
private:
    NeuralEvaluator& evaluator_;
    MCTSConfig config_;
    NodePool pool_;
    
    // Core MCTS phases
    
    // Phase 1: Select leaf node using UCB
    SelectResult select_leaf(Node* root);
    // Phase 1.5: If selected node is a node whose only move is a pass, expands this node
    // without needing the evaluation of the neural net
    void expand_pass_node(Node* node);
    
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
    void process_batch(
        const std::vector<SelectResult>& select_results,
        const std::vector<NNEval>& evals);

};

} // namespace OthelloMCTS

#endif // MCTS_SEARCH_H