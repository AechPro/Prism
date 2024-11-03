class Node(object):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

    def __init__(self, x, y, node_id, walkable, rng):
        self.x = x
        self.y = y
        self.id = node_id
        self.walkable = walkable
        self.occupied = False
        self.links = []
        self.visit_count = 0
        self.debug = False
        self.blue = False
        self.rng = rng

    def is_available(self):
        return self.walkable and not self.occupied

    def take_action(self, action, ignore_restrictions=False, laser_tag=False):
        if laser_tag:
            if action == Node.UP:
                action = 3  # up
            elif action == Node.RIGHT:
                action = 7  # right
            elif action == Node.DOWN:
                action = 5  # down
            elif action == Node.LEFT:
                action = 1  # left

        next_node = self
        # print("Node taking action", action, "of", len(self.links), "|", self.x, self.y)
        if len(self.links) > action >= 0:
            if ignore_restrictions:
                next_node = self.links[action]
            elif self.links[action].is_available():
                next_node = self.links[action]
        # print("Mapped to next node", next_node.x, "|", next_node.y)

        return next_node

    def get_adjacent(self):
        if len(self.links) < 9:
            return None
        left = self.links[7]
        right = self.links[1]
        up = self.links[3]
        down = self.links[5]
        return left, right, up, down

    def add_link(self, other):
        self.links.append(other)
