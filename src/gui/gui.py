import pygame
import threading
from othello_bindings import OthelloState, MCTSSearch

CELL_SIZE = 60
BOARD_SIZE = 8
WINDOW_SIZE = CELL_SIZE * BOARD_SIZE
BUTTON_HEIGHT = 80

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 128, 0)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
GRAY = (200, 200, 200)
LIGHT_GRAY = (220, 220, 220)
FPS = 30


class OthelloGUI:
    def __init__(self, model_path, human1=True, human2=False, simulations=400):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE + BUTTON_HEIGHT))
        pygame.display.set_caption("Othello - Human vs AI")
        self.clock = pygame.time.Clock()

        # Game state
        self.state = OthelloState()
        self.human_players = {1: human1, -1: human2}
        self.running = True
        self.history = []

        # AI setup
        self.mcts = MCTSSearch(model_path, num_simulations=simulations)
        self.ai_thinking = False
        self.ai_move = None
        self.ai_thread = None

        # UI elements
        self.font = pygame.font.SysFont(None, 24)
        self.small_font = pygame.font.SysFont(None, 20)
        
        # Buttons
        self.human_toggle_rect = pygame.Rect(10, WINDOW_SIZE + 5, 150, 30)
        self.takeback_rect = pygame.Rect(170, WINDOW_SIZE + 5, 100, 30)
        self.reset_rect = pygame.Rect(280, WINDOW_SIZE + 5, 80, 30)
        self.pass_rect = pygame.Rect(370, WINDOW_SIZE + 5, 80, 30)
        
        # Status text area
        self.status_rect = pygame.Rect(10, WINDOW_SIZE + 40, WINDOW_SIZE - 20, 35)

    def draw_buttons(self):
        # Human toggle button
        text = "H1 vs H2" if self.human_players[-1] else "H vs AI"
        color = LIGHT_GRAY if self.ai_thinking else GRAY
        pygame.draw.rect(self.screen, color, self.human_toggle_rect)
        pygame.draw.rect(self.screen, BLACK, self.human_toggle_rect, 2)
        self.screen.blit(
            self.font.render(text, True, BLACK),
            (self.human_toggle_rect.x + 15, self.human_toggle_rect.y + 5)
        )

        # Takeback button
        color = LIGHT_GRAY if (not self.history or self.ai_thinking) else GRAY
        pygame.draw.rect(self.screen, color, self.takeback_rect)
        pygame.draw.rect(self.screen, BLACK, self.takeback_rect, 2)
        self.screen.blit(
            self.font.render("Takeback", True, BLACK),
            (self.takeback_rect.x + 10, self.takeback_rect.y + 5)
        )
        
        # Reset button
        pygame.draw.rect(self.screen, GRAY, self.reset_rect)
        pygame.draw.rect(self.screen, BLACK, self.reset_rect, 2)
        self.screen.blit(
            self.font.render("Reset", True, BLACK),
            (self.reset_rect.x + 15, self.reset_rect.y + 5)
        )

        # Pass button
        is_pass_only = len(self.state.legal_moves()) == 1 and self.state.legal_moves()[0] == (-1, -1)
        color = GRAY if is_pass_only else LIGHT_GRAY
        pygame.draw.rect(self.screen, color, self.pass_rect)
        pygame.draw.rect(self.screen, BLACK, self.pass_rect, 2)
        self.screen.blit(
            self.font.render("Pass", True, BLACK),
            (self.pass_rect.x + 20, self.pass_rect.y + 5)
        )

    def draw_status(self):
        # Count pieces
        black_count = sum(1 for x in range(8) for y in range(8) if self.state.get_cell(x, y) == 1)
        white_count = sum(1 for x in range(8) for y in range(8) if self.state.get_cell(x, y) == -1)
        
        # Status message
        if self.state.is_terminal():
            result = self.state.game_result()
            if result == 1:
                status = "Game Over - Black Wins!"
            elif result == -1:
                status = "Game Over - White Wins!"
            else:
                status = "Game Over - Draw!"
        elif self.ai_thinking:
            status = "AI is thinking..."
        else:
            current = "Black" if self.state.current_player == 1 else "White"
            player_type = "Human" if self.human_players.get(self.state.current_player) else "AI"
            status = f"{current}'s turn ({player_type})"
        
        # Draw status
        pygame.draw.rect(self.screen, WHITE, self.status_rect)
        pygame.draw.rect(self.screen, BLACK, self.status_rect, 2)
        
        status_text = self.small_font.render(status, True, BLACK)
        score_text = self.small_font.render(f"Black: {black_count}  White: {white_count}", True, BLACK)
        
        self.screen.blit(status_text, (self.status_rect.x + 5, self.status_rect.y + 5))
        self.screen.blit(score_text, (self.status_rect.x + 5, self.status_rect.y + 20))

    def draw_board(self):
        self.screen.fill(WHITE)
        
        # Draw board background
        board_rect = pygame.Rect(0, 0, WINDOW_SIZE, WINDOW_SIZE)
        pygame.draw.rect(self.screen, GREEN, board_rect)
        
        # Draw grid and pieces
        for x in range(BOARD_SIZE):
            for y in range(BOARD_SIZE):
                rect = pygame.Rect(y * CELL_SIZE, x * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(self.screen, BLACK, rect, 1)
                
                piece = self.state.get_cell(x, y)
                if piece != 0:
                    color = BLACK if piece == 1 else WHITE
                    center = rect.center
                    pygame.draw.circle(self.screen, color, center, CELL_SIZE // 2 - 4)
                    if color == WHITE:
                        pygame.draw.circle(self.screen, BLACK, center, CELL_SIZE // 2 - 4, 2)

        # Highlight legal moves for human player
        if self.human_players.get(self.state.current_player) and not self.ai_thinking:
            for mx, my in self.state.legal_moves():
                if mx == -1:  # Pass move
                    continue
                rect = pygame.Rect(my * CELL_SIZE, mx * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                pygame.draw.circle(self.screen, BLUE, rect.center, 6)

        self.draw_buttons()
        self.draw_status()
        pygame.display.flip()

    def apply_move(self, x, y):
        """Apply a move and update tree for AI"""
        self.history.append(self.state)
        self.state = self.state.apply_move(x, y)
        
        # Inform MCTS about the move for tree reuse
        self.mcts.opponent_moved(x, y)

    def takeback(self):
        """Take back the last move"""
        if self.history and not self.ai_thinking:
            self.state = self.history.pop()
            self.mcts.reset()  # Reset AI tree after takeback
            self.draw_board()

    def reset_game(self):
        """Reset to a new game"""
        if not self.ai_thinking:
            self.state = OthelloState()
            self.history = []
            self.mcts.reset()
            self.draw_board()

    def ai_search_thread(self):
        """Run AI search in background thread"""
        try:
            x, y = self.mcts.get_move(self.state, temperature=0.1)
            self.ai_move = (x, y)
        except Exception as e:
            print(f"AI error: {e}")
            self.ai_move = None
        finally:
            self.ai_thinking = False

    def human_turn(self):
        """Handle human player input"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and not self.ai_thinking:
                mx, my = event.pos

                # Button clicks
                if self.human_toggle_rect.collidepoint(mx, my):
                    self.human_players[-1] = not self.human_players[-1]
                    self.mcts.reset()
                elif self.takeback_rect.collidepoint(mx, my):
                    self.takeback()
                elif self.reset_rect.collidepoint(mx, my):
                    self.reset_game()
                elif self.pass_rect.collidepoint(mx, my):
                    if (-1, -1) in self.state.legal_moves():
                        self.apply_move(-1, -1)
                elif my < WINDOW_SIZE:  # Board click
                    x = my // CELL_SIZE
                    y = mx // CELL_SIZE
                    if (x, y) in self.state.legal_moves():
                        self.apply_move(x, y)

    def ai_turn(self):
        """Handle AI player turn"""
        if not self.ai_thinking and self.ai_move is None:
            # Start AI search in background
            self.ai_thinking = True
            self.ai_thread = threading.Thread(target=self.ai_search_thread)
            self.ai_thread.start()
        
        # Check if AI has finished thinking
        if self.ai_move is not None:
            x, y = self.ai_move
            self.ai_move = None
            
            moves = self.state.legal_moves()
            if (x, y) in moves:
                self.apply_move(x, y)

    def run(self):
        """Main game loop"""
        while self.running:
            self.draw_board()

            if not self.state.is_terminal():
                current = self.state.current_player
                if self.human_players.get(current):
                    self.human_turn()
                else:
                    # Handle events even during AI turn (for quit)
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            self.running = False
                    self.ai_turn()
            else:
                # Game over - just handle quit
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False

            self.clock.tick(FPS)

        pygame.quit()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python othello_gui.py <model_path> [simulations]")
        sys.exit(1)
    
    model_path = sys.argv[1]
    simulations = int(sys.argv[2]) if len(sys.argv) > 2 else 400
    
    gui = OthelloGUI(model_path, human1=True, human2=False, simulations=simulations)
    gui.run()