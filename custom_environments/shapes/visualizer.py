import pygame


class ShapesVisualizer(object):
    def __init__(self, board_width, board_height, cell_size, desired_resolution=(1280, 720)):
        pygame.init()
        self.desired_resolution = desired_resolution
        self.res = (int(round(board_width * cell_size)),
                    int(round(board_height * cell_size)))

        self.cell_size = cell_size
        self.res_mod = (self.desired_resolution[0] / self.res[0], self.desired_resolution[1] / self.res[1])
        self.screen = pygame.display.set_mode(self.desired_resolution, pygame.RESIZABLE)

    def render(self, board):
        pygame.event.get()
        pygame.draw.rect(self.screen, (255, 255, 255), (0, 0,
                                                        int(round(self.res[0] * self.res_mod[0])),
                                                        int(round(self.res[1] * self.res_mod[1]))))

        rad_x = int(round(self.cell_size * self.res_mod[0]))
        rad_y = int(round(self.cell_size * self.res_mod[1]))
        for x in range(board.shape[0]):
            for y in range(board.shape[1]):
                nx = int(round(x * rad_x))
                ny = int(round(y * rad_y))

                if board[x, y] == 0:
                    color = (255, 0, 255)  # Pink.

                elif board[x, y] == 1:
                    color = (0, 255, 0)  # Green.

                elif board[x, y] == 2:
                    color = (0, 0, 255)  # Blue.

                elif board[x, y] == 3:
                    color = (255, 255, 0)  # Orange

                else:
                    color = (0, 0, 0)

                pygame.draw.rect(self.screen, color, (nx, ny, rad_x, rad_y))

        pygame.display.flip()

    def close(self):
        pygame.display.quit()
        pygame.quit()
