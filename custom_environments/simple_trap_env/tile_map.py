from . import Node


class TileMap(object):
    def __init__(self, node_map, rng):
        self.node_radius = 7
        self.nodes = []
        self.width = 0
        self.height = 0
        self.rng = rng
        self._load_map(node_map)

    def _build_links(self):
        for i in range(self.height):
            for j in range(self.width):
                target = self.nodes[i][j]
                for x in range(3):
                    for y in range(3):
                        tx = (j - 1) + x
                        ty = (i - 1) + y
                        if 0 <= tx < len(self.nodes[i]) and 0 <= ty < len(self.nodes):
                            n = self.nodes[ty][tx]
                            target.add_link(n)
                        else:
                            target.add_link(target)

    def _load_map(self, node_map):
        self.node_radius = node_map.node_radius
        self.nodes = []
        self._build_map(node_map.resolution[0], node_map.resolution[1])
        for line in node_map.lines:
            point1, point2 = line
            if point2[0] < point1[0]:
                t = point1
                point1 = point2
                point2 = t

            self._draw_line(point1, point2)
        self._build_links()

    def _build_map(self, res_x, res_y):
        n_id = 0
        self.width = 0
        self.height = 0
        for y in range(0, res_y, self.node_radius):
            row = []
            self.height += 1
            self.width = 0
            for x in range(0, res_x, self.node_radius):
                node = Node(x, y, n_id, True, self.rng)
                row.append(node)
                n_id += 1
                self.width += 1
            self.nodes.append(row)

    def _draw_line(self, point1, point2):
        x1, y1 = point1
        x2, y2 = point2
        if x1 == x2:
            start_y = min(y1, y2)
            end_y = max(y1, y2)
            for y in range(start_y, end_y):
                node = self.get_node(x1, y)
                if node is not None:
                    node.walkable = False
        else:
            m = (y2-y1)/(x2-x1)
            b = y2 - x2 * m
            for x in range(x1, x2):
                y = m*x + b
                node = self.get_node(x, y)
                if node is not None:
                    node.walkable = False

    def get_node(self, x, y):
        x = int(round(x / self.node_radius))
        y = int(round(y / self.node_radius))
        if len(self.nodes[0]) > x >= 0 and len(self.nodes) > y >= 0:
            return self.nodes[y][x]
        return None

    def cleanup(self):
        del self.nodes
        self.nodes = None
