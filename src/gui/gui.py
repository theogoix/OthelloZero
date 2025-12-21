# gui.py
import pygame
from othello_bindings import OthelloMove
from othello_engine_stub import OthelloState, OthelloEngine, OthelloMove

CELL_SIZE = 60
BOARD_SIZE = 8
WINDOW_SIZE = CELL_SIZE * BOARD_SIZE

BLACK = (0,0,0)
WHITE = (255,255,255)
GREEN = (0,128,0)
BLUE = (0,0,255)

class OthelloGUI:
    def __init__(self, engine):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
        pygame.display.set_caption("Othello")
        self.clock = pygame.time.Clock()
        self.engine = engine
        self.state = OthelloState()
        self.running = True
        self.human_player = 1  # 1 = black, -1 = white
    
    def draw_board(self):
        self.screen.fill(GREEN)
        for x in range(BOARD_SIZE):
            for y in range(BOARD_SIZE):
                rect = pygame.Rect(y*CELL_SIZE, x*CELL_SIZE, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(self.screen, BLACK, rect, 1)
                piece = self.state.board[x][y]
                if piece != 0:
                    color = BLACK if piece == 1 else WHITE
                    pygame.draw.circle(self.screen, color, rect.center, CELL_SIZE//2 - 4)
        pygame.display.flip()

    def apply_move(self, move: OthelloMove):
        self.state.board[move.x][move.y] = self.state.current_player
        self.state.current_player *= -1  # switch player

    def run(self):
        while self.running:
            self.draw_board()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and self.state.current_player == self.human_player:
                    mx, my = event.pos
                    move = OthelloMove(mx//CELL_SIZE, my//CELL_SIZE)
                    if self.state.board[move.x][move.y] == 0:
                        self.apply_move(move)
            
            # AI move
            if self.state.current_player != self.human_player:
                move = self.engine.selectMove(self.state)
                if move:
                    self.apply_move(move)
            
            self.clock.tick(10)

if __name__ == "__main__":
    engine = OthelloEngine()
    gui = OthelloGUI(engine)
    gui.run()
