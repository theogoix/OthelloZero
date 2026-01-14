#include "engine/mcts/mcts_search.h"
#include <cmath>
#include <algorithm>
#include <random>
#include <iostream>

namespace OthelloMCTS {

MCTSSearch::MCTSSearch(NeuralEvaluator& evaluator, const MCTSConfig& config)
    : evaluator_(evaluator), config_(config) {}

Othello::OthelloMove MCTSSearch::search(const Othello::OthelloState& root_state) {
    // Create root node
    root_ = pool_.allocate(root_state);
    
    int simulations_done = 0;
    
    while (simulations_done < config_.num_simulations) {
        // Phase 1: Collect batch of leaf nodes
        std::vector<Node*> batch_nodes;
        std::vector<SelectResult> batch_results;
        
        int batch_size = std::min(
            config_.batch_size,
            config_.num_simulations - simulations_done
        );
        
        for (int i = 0; i < batch_size; i++) {
            SelectResult result = select_leaf(root_);
            
            // If leaf is terminal, handle immediately
            if (result.leaf->is_terminal()) {
                float terminal_value = Othello::OthelloOps::gameResult(result.leaf->state);
                backpropagate(result, terminal_value);
                simulations_done++;
                continue;
            }
            
            batch_nodes.push_back(result.leaf);
            batch_results.push_back(result);
        }
        
        if (batch_nodes.empty()) continue;
        
        // Phase 2: Evaluate batch with neural network
        std::vector<Othello::OthelloState> states;
        for (Node* node : batch_nodes) {
            states.push_back(node->state);
        }
        
        auto evaluations = evaluator_.evaluate_batch(states);
        
        // Phase 3: Expand nodes and backpropagate
        for (size_t i = 0; i < batch_nodes.size(); i++) {
            Node* node = batch_nodes[i];
            
            if (!node->expanded) {
                expand_node(node, evaluations[i]);
            }
            
            backpropagate(batch_results[i], evaluations[i].value);
        }
        
        simulations_done += batch_nodes.size();
        
        // Add Dirichlet noise to root after first batch (when root is expanded)
        if (simulations_done == batch_size && 
            config_.use_dirichlet && 
            root_->expanded) {
            add_dirichlet_noise(root_);
        }
    }
    
    // Select best move based on visit counts
    if (root_->edges.empty()) {
        return -1;  // No legal moves
    }
    
    auto best_edge = std::max_element(
        root_->edges.begin(),
        root_->edges.end(),
        [](const Edge& a, const Edge& b) {
            return a.visit_count < b.visit_count;
        }
    );
    
    // Print some statistics
    std::cout << "MCTS Statistics:\n";
    std::cout << "  Simulations: " << simulations_done << "\n";
    std::cout << "  Nodes created: " << pool_.size() << "\n";
    std::cout << "  Root visits: " << root_->visit_count << "\n";
    std::cout << "  Root value: " << root_->Q() << "\n";
    std::cout << "  Best move: " << best_edge->move 
              << " (visits: " << best_edge->visit_count 
              << ", Q: " << best_edge->Q() << ")\n";
    
    return best_edge->move;
}

SelectResult MCTSSearch::select_leaf(Node* root) {
    SelectResult result;
    Node* node = root;
    
    while (node->expanded && !node->is_terminal()) {
        Edge* best = select_best_edge(node);
        
        if (!best) break;  // No valid edges (shouldn't happen if expanded)
        
        // Add virtual loss for batch diversity
        best->visit_count++;
        best->total_value -= config_.virtual_loss;
        
        // Record the decision
        result.path.push_back({node, best});
        
        // Traverse (lazy child creation)
        node = get_or_create_child(node, best);
    }
    
    result.leaf = node;
    return result;
}

Node* MCTSSearch::get_or_create_child(Node* parent, Edge* edge) {
    if (edge->child == nullptr) {
        // Lazy child creation
        Othello::OthelloState child_state = Othello::OthelloOps::applyMove(
            parent->state, edge->move);
        edge->child = pool_.allocate(child_state);
    }
    return edge->child;
}

void MCTSSearch::expand_node(Node* node, const NNEval& eval) {
    // Legal moves are already precomputed in the state!
    uint64_t legal_moves_bitboard = node->state.legalMoves;
    
    if (legal_moves_bitboard == 0) {
        // Terminal node - no children
        node->expanded = true;
        return;
    }
    
    // Convert bitboard to move list
    auto legal_moves = bitboard_to_moves(legal_moves_bitboard);
    
    // Normalize policy over legal moves
    float policy_sum = 0.0f;
    for (auto move : legal_moves) {
        policy_sum += eval.policy[move];
    }
    
    if (policy_sum < 1e-6f) policy_sum = 1.0f;  // Avoid division by zero
    
    // Create edges for all legal moves (lazy child creation - no nodes yet!)
    node->edges.reserve(legal_moves.size());
    for (auto move : legal_moves) {
        Edge edge;
        edge.move = move;
        edge.prior = eval.policy[move] / policy_sum;  // Normalized prior
        edge.child = nullptr;  // Lazy creation - will be created on first traversal
        
        node->edges.push_back(edge);
    }
    
    node->expanded = true;
}

void MCTSSearch::backpropagate(const SelectResult& result, float value) {
    // Update leaf node
    result.leaf->visit_count++;
    result.leaf->total_value += value;
    value = -value;  // Flip perspective for parent
    
    // Update path in reverse (from leaf to root)
    for (auto it = result.path.rbegin(); it != result.path.rend(); ++it) {
        // Update node statistics
        it->node->visit_count++;
        it->node->total_value += value;
        
        // Update edge statistics (remove virtual loss, add real value)
        it->edge->total_value += config_.virtual_loss + value;
        // Note: visit_count was already incremented during selection
        
        // Flip value for next level up
        value = -value;
    }
}

float MCTSSearch::ucb_score(const Edge& edge, int parent_visits) const {
    float Q = edge.Q();
    float U = config_.c_puct * edge.prior * 
              std::sqrt((float)parent_visits) / (1.0f + edge.visit_count);
    return Q + U;
}

Edge* MCTSSearch::select_best_edge(Node* node) const {
    if (node->edges.empty()) return nullptr;
    
    Edge* best = nullptr;
    float best_score = -1e9f;
    
    for (auto& edge : node->edges) {
        float score = ucb_score(edge, node->visit_count);
        if (score > best_score) {
            best_score = score;
            best = &edge;
        }
    }
    
    return best;
}

std::vector<Othello::OthelloMove> MCTSSearch::bitboard_to_moves(uint64_t bitboard) const {
    std::vector<Othello::OthelloMove> moves;
    
    for (int pos = 0; pos < 64; pos++) {
        if (bitboard & (1ULL << pos)) {
            moves.push_back(pos);
        }
    }
    
    return moves;
}

void MCTSSearch::add_dirichlet_noise(Node* root) {
    if (root->edges.empty()) return;
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::gamma_distribution<float> gamma(config_.dirichlet_alpha, 1.0f);
    
    // Generate Dirichlet noise
    std::vector<float> noise;
    float noise_sum = 0.0f;
    for (size_t i = 0; i < root->edges.size(); i++) {
        float n = gamma(gen);
        noise.push_back(n);
        noise_sum += n;
    }
    
    // Normalize and mix with priors
    for (size_t i = 0; i < root->edges.size(); i++) {
        noise[i] /= noise_sum;
        root->edges[i].prior = 
            (1.0f - config_.dirichlet_epsilon) * root->edges[i].prior +
            config_.dirichlet_epsilon * noise[i];
    }
}

std::vector<std::pair<Othello::OthelloMove, int>> 
MCTSSearch::get_move_visits() const {
    std::vector<std::pair<Othello::OthelloMove, int>> result;
    if (!root_) return result;
    
    for (const auto& edge : root_->edges) {
        result.emplace_back(edge.move, edge.visit_count);
    }
    
    return result;
}

void MCTSSearch::reset() {
    pool_.reset();
    root_ = nullptr;
}

} // namespace OthelloMCTS