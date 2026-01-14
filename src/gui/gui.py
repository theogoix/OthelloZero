import pygame
import copy
from othello_bindings import OthelloState

CELL_SIZE = 60
BOARD_SIZE = 8
WINDOW_SIZE = CELL_SIZE * BOARD_SIZE
BUTTON_HEIGHT = 40

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 128, 0)
BLUE = (0, 0, 255)
GRAY = (200, 200, 200)
FPS = 30


class OthelloGUI:
    def __init__(self, human1=True, human2=False):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE + BUTTON_HEIGHT))
        pygame.display.set_caption("Othello")
        self.clock = pygame.time.Clock()

        # Game state
        self.state = OthelloState()
        self.human_players = {1: human1, -1: human2}  # True = human, False = AI
        self.running = True
        self.history = []

        # Buttons
        self.font = pygame.font.SysFont(None, 24)
        self.human_toggle_rect = pygame.Rect(10, WINDOW_SIZE + 5, 150, 30)
        self.takeback_rect = pygame.Rect(200, WINDOW_SIZE + 5, 100, 30)

    def draw_buttons(self):
        # Human toggle button
        text = "H1 vs H2" if self.human_players[-1] else "H vs AI"
        pygame.draw.rect(self.screen, GRAY, self.human_toggle_rect)
        self.screen.blit(self.font.render(text, True, BLACK),
                         (self.human_toggle_rect.x + 5, self.human_toggle_rect.y + 5))

        # Takeback button
        pygame.draw.rect(self.screen, GRAY, self.takeback_rect)
        self.screen.blit(self.font.render("Takeback", True, BLACK),
                         (self.takeback_rect.x + 5, self.takeback_rect.y + 5))

    def draw_board(self):
        self.screen.fill(GREEN)
        for x in range(BOARD_SIZE):
            for y in range(BOARD_SIZE):
                rect = pygame.Rect(y * CELL_SIZE, x * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(self.screen, BLACK, rect, 1)
                piece = self.state.get_cell(x, y)
                if piece != 0:
                    color = BLACK if piece == 1 else WHITE
                    pygame.draw.circle(self.screen, color, rect.center, CELL_SIZE // 2 - 4)

        # Highlight legal moves for human
        if self.human_players.get(self.state.current_player):
            for mx, my in self.state.legal_moves():
                if mx == -1:
                    continue
                rect = pygame.Rect(my * CELL_SIZE, mx * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                pygame.draw.circle(self.screen, BLUE, rect.center, 6)

        self.draw_buttons()
        pygame.display.flip()

    def apply_move(self, x, y):
        self.history.append(self.state)         # store current state in history
        self.state = self.state.apply_move(x, y)  # returns a new PyOthelloState

    def takeback(self):
        if self.history:
            self.state = self.history.pop()      # just restore the previous state
            self.draw_board()
    
    def human_turn(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = event.pos

                # Button clicks
                if self.human_toggle_rect.collidepoint(mx, my):
                    self.human_players[-1] = not self.human_players[-1]
                elif self.takeback_rect.collidepoint(mx, my):
                    self.takeback()
                else:
                    # Board click
                    x = my // CELL_SIZE
                    y = mx // CELL_SIZE
                    if (x, y) in self.state.legal_moves():
                        self.apply_move(x, y)

    def ai_turn(self):
        moves = self.state.legal_moves()
        if not moves:
            return
        # Simple AI: pick first legal move
        x, y = moves[0]
        self.apply_move(x, y)

    def run(self):
        while self.running and not self.state.is_terminal():
            self.draw_board()

            current = self.state.current_player
            if self.human_players.get(current):
                self.human_turn()
            else:
                self.ai_turn()

            self.clock.tick(FPS)

        self.draw_board()
        print("Game Over!")
        result = self.state.game_result()
        if result == 1:
            print("Black wins!")
        elif result == -1:
            print("White wins!")
        else:
            print("Draw!")

        pygame.quit()


if __name__ == "__main__":
    gui = OthelloGUI(human1=True, human2=False)  # H vs AI by default
    gui.run()
