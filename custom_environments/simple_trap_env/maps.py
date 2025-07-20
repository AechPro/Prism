

class Map(object):
    def __init__(self):
        self.resolution = (1280, 720)
        self.node_radius = 6
        self.lines = []

    def _add_line(self, point1, point2):
        self.lines.append((point1, point2))


class SimpleTrapMap(Map):
    def __init__(self):
        super().__init__()
        trap_gap = 200
        right_wall_x = 780
        left_wall_x = 500
        left_right_dist = right_wall_x - left_wall_x
        gap_right_x = left_wall_x + left_right_dist//2 + trap_gap//2
        gap_left_x = left_wall_x + left_right_dist//2 - trap_gap//2

        wall_height = 630
        wall_top = 50
        wall_bottom = wall_top + wall_height

        # Right half of cage
        self._add_line((right_wall_x, wall_top), (right_wall_x, wall_bottom))
        self._add_line((right_wall_x, wall_top), (gap_right_x, wall_top))
        self._add_line((right_wall_x, wall_bottom), (gap_right_x, wall_bottom))

        # Left half of cage
        self._add_line((left_wall_x, wall_top), (left_wall_x, wall_bottom))
        self._add_line((left_wall_x, wall_top), (gap_left_x, wall_top))
        self._add_line((left_wall_x, wall_bottom), (gap_left_x, wall_bottom))

