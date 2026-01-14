#ifndef NEURAL_EVALUATOR_H
#define NEURAL_EVALUATOR_H

#include <vector>
#include <array>
#include <string>
#include "othello/othello.h"

#ifdef USE_TORCH
#include <torch/script.h>
#include <torch/torch.h>
#endif

namespace OthelloMCTS {

// Result of neural network evaluation
struct NNEval {
    std::array<float, 64> policy;  // Move probabilities for each square
    float value;                    // Position evaluation [-1, 1]
};

class NeuralEvaluator {
public:
    NeuralEvaluator(const std::string& model_path, bool use_gpu = true);
    ~NeuralEvaluator() = default;
    
    // Evaluate a batch of positions
    std::vector<NNEval> evaluate_batch(
        const std::vector<Othello::OthelloState>& states);
    
    bool is_available() const { return initialized_; }

private:
    bool initialized_ = false;
    
#ifdef USE_TORCH
    torch::jit::script::Module module_;
    torch::Device device_;
    
    // Convert board states to neural net input tensor
    torch::Tensor states_to_tensor(
        const std::vector<Othello::OthelloState>& states);
    
    // Encode single state into tensor slice
    void encode_state(const Othello::OthelloState& state, 
                     torch::Tensor tensor_slice);
#endif
};

// Fallback: random evaluator for testing without NN
class RandomEvaluator {
public:
    std::vector<NNEval> evaluate_batch(
        const std::vector<Othello::OthelloState>& states);
};

} // namespace OthelloMCTS

#endif // NEURAL_EVALUATOR_H