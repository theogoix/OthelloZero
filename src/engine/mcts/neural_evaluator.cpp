#include "neural_evaluator.h"
#include <iostream>
#include <random>

namespace OthelloMCTS {



NeuralEvaluator::NeuralEvaluator(const std::string& model_path, bool use_gpu)
    : device_(use_gpu && torch::cuda::is_available() ? torch::kCUDA : torch::kCPU) {
    
    try {
        module_ = torch::jit::load(model_path);
        module_.to(device_);
        module_.eval();
        initialized_ = true;
        
        std::cout << "Neural network loaded successfully\n";
        std::cout << "Using device: " 
                  << (device_.is_cuda() ? "CUDA" : "CPU") << "\n";
    } catch (const c10::Error& e) {
        std::cerr << "Error loading model: " << e.what() << "\n";
        initialized_ = false;
    }
}

torch::Tensor NeuralEvaluator::states_to_tensor(
    const std::vector<Othello::OthelloState>& states) {
    
    int batch_size = states.size();
    
    // Input: [batch, 3, 8, 8]
    // Channel 0: Current player's discs
    // Channel 1: Opponent's discs
    // Channel 2: Legal moves
    auto tensor = torch::zeros({batch_size, 3, 8, 8});
    
    for (int b = 0; b < batch_size; b++) {
        encode_state(states[b], tensor[b]);
    }
    
    return tensor;
}

void NeuralEvaluator::encode_state(const Othello::OthelloState& state,
                                   torch::Tensor tensor_slice) {

    // Fill channels
    for (int row = 0; row < 8; row++) {
        for (int col = 0; col < 8; col++) {
            int pos = row * 8 + col;
            uint64_t mask = 1ULL << pos;
            
            // Channel 0: Current player's discs
            if (state.currentDiscs & mask) {
                tensor_slice[0][row][col] = 1.0f;
            }
            
            // Channel 1: Opponent's discs
            if (state.opponentDiscs & mask) {
                tensor_slice[1][row][col] = 1.0f;
            }
            
            // Channel 2: Legal moves
            if (state.opponentDiscs & mask) {
                tensor_slice[1][row][col] = 1.0f;
            }
        }
    }
}

std::vector<NNEval> NeuralEvaluator::evaluate_batch(
    const std::vector<Othello::OthelloState>& states) {
    
    if (!initialized_) {
        throw std::runtime_error("Neural network not initialized");
    }
    
    // Convert states to tensor
    torch::Tensor input = states_to_tensor(states);
    input = input.to(device_);
    
    // Run inference
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input);
    
    torch::NoGradGuard no_grad;
    auto output = module_.forward(inputs);
    
    // Parse output - expect tuple of (policy, value)
    auto output_tuple = output.toTuple();
    torch::Tensor policy_tensor = output_tuple->elements()[0].toTensor();
    torch::Tensor value_tensor = output_tuple->elements()[1].toTensor();
    
    // Move to CPU for processing
    policy_tensor = policy_tensor.cpu();
    value_tensor = value_tensor.cpu();
    
    // Convert to results
    std::vector<NNEval> results;
    results.reserve(states.size());
    
    for (size_t i = 0; i < states.size(); i++) {
        NNEval eval;
        
        // Extract policy (shape: [batch, 64])
        auto policy_accessor = policy_tensor.accessor<float, 2>();
        for (int j = 0; j < 64; j++) {
            eval.policy[j] = policy_accessor[i][j];
        }
        
        // Extract value (shape: [batch, 1] or [batch])
        auto value_accessor = value_tensor.accessor<float, 2>();
        eval.value = value_accessor[i][0];
        
        results.push_back(eval);
    }
    
    return results;
}




// Random evaluator for testing
std::vector<NNEval> RandomEvaluator::evaluate_batch(
    const std::vector<Othello::OthelloState>& states) {
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    
    std::vector<NNEval> results;
    results.reserve(states.size());
    
    for (const auto& state : states) {
        NNEval eval;
        
        // Random policy
        for (int i = 0; i < 64; i++) {
            eval.policy[i] = dist(gen);
        }
        
        // Random value between -1 and 1
        eval.value = dist(gen) * 2.0f - 1.0f;
        
        results.push_back(eval);
    }
    
    return results;
}

} // namespace OthelloMCTS