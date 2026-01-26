// py_othello_state.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "othello/othello.h"
#include "engine/mcts/mcts_search.h"

namespace py = pybind11;
using namespace Othello;
using namespace OthelloMCTS;

class PyOthelloState {
public:
    OthelloState state;
    int current_player;  // 1 = currentDiscs player, -1 = opponentDiscs player

    PyOthelloState()
        : state(OthelloOps::initialState()), current_player(1) {}

    // Return legal moves as (x,y) tuples, PASS -> (-1,-1)
    std::vector<std::tuple<int,int>> legal_moves() const {
        std::vector<OthelloMove> moves = OthelloOps::generateMoves(state);
        std::vector<std::tuple<int,int>> coords;
        for (auto m : moves) {
            if (m == PASS)
                coords.emplace_back(-1, -1);
            else
                coords.emplace_back(m / 8, m % 8);
        }
        return coords;
    }

    // Apply move given coordinates
    PyOthelloState apply_move(int x, int y) const {
        OthelloMove move = (x < 0 || y < 0) ? PASS : static_cast<OthelloMove>(x*8 + y);
        OthelloState new_state = OthelloOps::applyMove(state, move);

        PyOthelloState new_py_state;
        new_py_state.state = new_state;
        new_py_state.current_player = -current_player;  // switch player

        return new_py_state;
    }

    bool is_terminal() const { return OthelloOps::isTerminal(state); }

    int game_result() const { return current_player * static_cast<int>(OthelloOps::gameResult(state)); }

    uint64_t current_discs() const { return state.currentDiscs; }
    uint64_t opponent_discs() const { return state.opponentDiscs; }
    bool lastMoveWasPass() const { return state.lastMoveWasPass; }

    // Return cell content relative to current player: 1 = current player disc, -1 = opponent, 0 = empty
    int get_cell(int x, int y) const {
        uint64_t mask = 1ULL << (x*8 + y);
        if (state.currentDiscs & mask) return current_player;
        if (state.opponentDiscs & mask) return -current_player;
        return 0;
    }

    // Return full board as 8x8 Python-friendly list
    std::vector<std::vector<int>> get_board() const {
        std::vector<std::vector<int>> board(8, std::vector<int>(8, 0));
        for (int x = 0; x < 8; x++)
            for (int y = 0; y < 8; y++)
                board[x][y] = get_cell(x, y);
        return board;
    }
};


// Wrapper for MCTS search
class PyMCTSSearch {
private:
    std::unique_ptr<NeuralEvaluator> evaluator_;
    std::unique_ptr<MCTSSearch> mcts_;
    Node* current_root_;
    
public:
    PyMCTSSearch(const std::string& model_path, int num_simulations = 800) {
        // Initialize neural evaluator with your trained model
        evaluator_ = std::make_unique<NeuralEvaluator>(model_path);
        
        // Configure MCTS
        MCTSConfig config;
        config.num_simulations = num_simulations;
        config.c_puct = 1.5f;
        config.use_dirichlet = false;  // Disable for play mode
        
        mcts_ = std::make_unique<MCTSSearch>(*evaluator_, config);
        current_root_ = nullptr;
    }
    
    // Get best move for current state
    std::tuple<int, int> get_move(const PyOthelloState& py_state, float temperature = 0.1f) {
        // Run MCTS search
        Node* root = mcts_->search(py_state.state, current_root_);
        
        // Select move with temperature
        OthelloMove move = mcts_->select_move(root, temperature);
        
        // Update root for tree reuse
        current_root_ = mcts_->get_child_node(root, move);
        
        // Convert move to coordinates
        if (move == PASS) {
            return std::make_tuple(-1, -1);
        }
        return std::make_tuple(move / 8, move % 8);
    }
    
    // Get policy distribution (for analysis)
    std::vector<std::tuple<int, int, float>> get_policy(const PyOthelloState& py_state) {
        Node* root = mcts_->search(py_state.state, current_root_);
        auto policy = mcts_->get_policy(root);
        
        std::vector<std::tuple<int, int, float>> result;
        for (int i = 0; i < 64; i++) {
            if (policy[i] > 0.0f) {
                result.emplace_back(i / 8, i % 8, policy[i]);
            }
        }
        
        return result;
    }
    
    // Get visit counts (for analysis)
    std::vector<std::tuple<int, int, int>> get_visit_counts(const PyOthelloState& py_state) {
        Node* root = mcts_->search(py_state.state, current_root_);
        auto visits = mcts_->get_visit_counts(root);
        
        std::vector<std::tuple<int, int, int>> result;
        for (int i = 0; i < 64; i++) {
            if (visits[i] > 0) {
                result.emplace_back(i / 8, i % 8, visits[i]);
            }
        }
        
        return result;
    }
    
    // Reset tree (call when starting new game or after opponent move)
    void reset() {
        mcts_->reset();
        current_root_ = nullptr;
    }
    
    // Inform search about opponent's move (for tree reuse)
    void opponent_moved(int x, int y) {
        if (current_root_ != nullptr) {
            OthelloMove move = (x < 0 || y < 0) ? PASS : static_cast<OthelloMove>(x * 8 + y);
            current_root_ = mcts_->get_child_node(current_root_, move);
        }
    }
    
    // Get statistics
    int get_nodes_created() const {
        return mcts_->get_nodes_created();
    }
};





PYBIND11_MODULE(othello_bindings, m) {
    py::class_<PyOthelloState>(m, "OthelloState")
        .def(py::init<>())
        .def("legal_moves", &PyOthelloState::legal_moves)
        .def("apply_move", &PyOthelloState::apply_move)
        .def("is_terminal", &PyOthelloState::is_terminal)
        .def("game_result", &PyOthelloState::game_result)
        .def("get_cell", &PyOthelloState::get_cell)
        .def("get_board", &PyOthelloState::get_board)
        .def_readonly("current_player", &PyOthelloState::current_player)
        .def_property_readonly("current_discs", &PyOthelloState::current_discs)
        .def_property_readonly("opponent_discs", &PyOthelloState::opponent_discs)
        .def_property_readonly("lastMoveWasPass", &PyOthelloState::lastMoveWasPass)
        ;
    
    py::class_<PyMCTSSearch>(m, "MCTSSearch")
        .def(py::init<const std::string&, int>(), 
             py::arg("model_path"), 
             py::arg("num_simulations") = 800)
        .def("get_move", &PyMCTSSearch::get_move, 
             py::arg("state"), 
             py::arg("temperature") = 0.1f)
        .def("get_policy", &PyMCTSSearch::get_policy)
        .def("get_visit_counts", &PyMCTSSearch::get_visit_counts)
        .def("reset", &PyMCTSSearch::reset)
        .def("opponent_moved", &PyMCTSSearch::opponent_moved)
        .def("get_nodes_created", &PyMCTSSearch::get_nodes_created);
}
