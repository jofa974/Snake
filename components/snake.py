import math

from ui import BASE_SIZE, BASE_SPEED, HEIGHT, WIDTH, X_GRID, Y_GRID


def norm_speed(speed):
    return math.floor(speed / BASE_SPEED)


def speed_sign(speed):
    return int(math.copysign(1, norm_speed(speed)))


class Snake:
    """Snake player"""

    def __init__(self, x=int(X_GRID / 2), y=int(Y_GRID / 2), s=(BASE_SPEED, 0)):
        self.speed = s
        self.dead = False
        self.body_list = [(x * BASE_SIZE, y * BASE_SIZE,)]

        for i in range(1, 3):
            self.body_list.append((int(WIDTH / 2) - i * BASE_SPEED, int(HEIGHT / 2)))

    def update(self):
        for i in range(len(self.body_list) - 1, 0, -1):
            self.body_list[i] = self.body_list[i - 1]

    def move(self):
        self.update()
        self.body_list[0] = tuple(
            self.body_list[0][i] + self.speed[i] for i in range(2)
        )

    def detect_collisions(self):
        # Did this update cause us to hit a wall?
        if self.is_collision_wall(self.get_position(0)):
            self.dead = True

        # Did this update cause us to hit a body part
        if self.is_collision_body(self.get_position(0)):
            self.dead = True

    def eat(self, target):
        """returns true if the snake collides with the target"""
        head_pos = self.get_position(0)
        return head_pos[0] == target[0] and head_pos[1] == target[1]

    def change_direction(self, key):
        if key == "up" and self.speed[1] == 0:
            self.speed = (0, -BASE_SPEED)
        elif key == "down" and self.speed[1] == 0:
            self.speed = (0, BASE_SPEED)
        elif key == "left" and self.speed[0] == 0:
            self.speed = (-BASE_SPEED, 0)
        elif key == "right" and self.speed[0] == 0:
            self.speed = (BASE_SPEED, 0)

    def grow(self):
        tail = self.body_list[-1]
        self.body_list.append(tail)

    def get_position(self, idx):
        return (
            int(self.body_list[idx][0] / BASE_SIZE),
            int(self.body_list[idx][1] / BASE_SIZE),
        )

    def get_neighbors(self, position, direction):
        """Get the coordinates of the nearest neighbors: in front, to the left and the right"""
        neighbors = {}
        if self.speed == (BASE_SPEED, 0):
            neighbors["front"] = (position[0] + 1, position[1])
            neighbors["left"] = (position[0], position[1] - 1)
            neighbors["right"] = (position[0], position[1] + 1)
        if self.speed == (-BASE_SPEED, 0):
            neighbors["front"] = (position[0] - 1, position[1])
            neighbors["left"] = (position[0], position[1] + 1)
            neighbors["right"] = (position[0], position[1] - 1)
        if self.speed == (0, BASE_SPEED):
            neighbors["front"] = (position[0], position[1] + 1)
            neighbors["left"] = (position[0] + 1, position[1])
            neighbors["right"] = (position[0] - 1, position[1])
        if self.speed == (0, -BASE_SPEED):
            neighbors["front"] = (position[0], position[1] - 1)
            neighbors["left"] = (position[0] - 1, position[1])
            neighbors["right"] = (position[0] + 1, position[1])
        return neighbors

    def get_next_key(self, way):
        if way == "front":
            return "front", self.speed
        if way == "left":
            if self.speed == (BASE_SPEED, 0):
                return "up", (0, -BASE_SPEED)
            if self.speed == (-BASE_SPEED, 0):
                return "down", (0, BASE_SPEED)
            if self.speed == (0, BASE_SPEED):
                return "left", (-BASE_SPEED, 0)
            if self.speed == (0, -BASE_SPEED):
                return "right", (BASE_SPEED, 0)
        if way == "right":
            if self.speed == (BASE_SPEED, 0):
                return "down", (0, BASE_SPEED)
            if self.speed == (-BASE_SPEED, 0):
                return "up", (0, -BASE_SPEED)
            if self.speed == (0, BASE_SPEED):
                return "right", (BASE_SPEED, 0)
            if self.speed == (0, -BASE_SPEED):
                return "left", (-BASE_SPEED, 0)

    def is_collision_wall(self, pos):
        x_head, y_head = pos
        return (
            x_head == 0 or x_head == X_GRID - 1 or y_head == 0 or y_head == Y_GRID - 1
        )

    def get_body_position_list(self):
        to_return = []
        for idx in range(0, len(self.body_list)):
            to_return.append(self.get_position(idx))
        return to_return

    def is_collision_body(self, pos):
        x_head, y_head = pos
        body_positions = self.get_body_position_list()
        for body_pos in body_positions[1:]:
            if body_pos[0] == x_head and body_pos[1] == y_head:
                return True
        return False

    def get_distance_to_target(self, snake_pos, target_pos, norm=2):
        dist = sum([pow(abs(snake_pos[i] - target_pos[i]), norm) for i in range(2)])
        dist = pow(dist, 1.0 / norm)
        return dist

    def is_clear_ahead(self):
        snake_pos = self.get_position(0)
        speed = self.speed
        next_pos = [snake_pos[i] + speed[i] for i in range(2)]
        return not self.is_collision_wall(next_pos) and not self.is_collision_body(
            next_pos
        )

    def get_next_pos_left(self):
        snake_pos = self.get_position(0)
        if self.speed[0] > 0:
            return (snake_pos[0], snake_pos[1] - 1)
        elif self.speed[0] < 0:
            return (snake_pos[0], snake_pos[1] + 1)
        elif self.speed[1] > 0:
            return (snake_pos[0] + 1, snake_pos[1])
        elif self.speed[1] < 0:
            return (snake_pos[0] - 1, snake_pos[1])

    def get_next_pos_right(self):
        snake_pos = self.get_position(0)
        if self.speed[0] > 0:
            return (snake_pos[0], snake_pos[1] + 1)
        elif self.speed[0] < 0:
            return (snake_pos[0], snake_pos[1] - 1)
        elif self.speed[1] > 0:
            return (snake_pos[0] - 1, snake_pos[1])
        elif self.speed[1] < 0:
            return (snake_pos[0] + 1, snake_pos[1])

    def is_clear_left(self):
        next_pos = self.get_next_pos_left()
        return not self.is_collision_wall(next_pos) and not self.is_collision_body(
            next_pos
        )

    def is_clear_right(self):
        next_pos = self.get_next_pos_right()
        return not self.is_collision_wall(next_pos) and not self.is_collision_body(
            next_pos
        )

    def is_food_ahead(self, apple_pos):
        snake_pos = self.get_position(0)
        current_dist = self.get_distance_to_target(snake_pos, apple_pos)

        speed = self.speed
        next_pos = tuple([snake_pos[i] + int(speed[i] / BASE_SPEED) for i in range(2)])
        next_dist = self.get_distance_to_target(next_pos, apple_pos)

        return next_dist < current_dist

    def is_food_left(self, apple_pos):
        snake_pos = self.get_position(0)
        if self.speed[0] > 0:
            return apple_pos[1] <= snake_pos[1]
        if self.speed[0] < 0:
            return apple_pos[1] >= snake_pos[1]
        if self.speed[1] > 0:
            return apple_pos[0] >= snake_pos[0]
        if self.speed[1] < 0:
            return apple_pos[0] <= snake_pos[0]

    def is_food_right(self, apple_pos):
        snake_pos = self.get_position(0)
        if self.speed[0] > 0:
            return apple_pos[1] >= snake_pos[1]
        if self.speed[0] < 0:
            return apple_pos[1] <= snake_pos[1]
        if self.speed[1] > 0:
            return apple_pos[0] <= snake_pos[0]
        if self.speed[1] < 0:
            return apple_pos[0] >= snake_pos[0]

    def get_distance_to_north_wall(self, norm=2):
        snake_pos = self.get_position(0)
        return self.get_distance_to_target(snake_pos, [snake_pos[0], 0])

    def get_distance_to_south_wall(self, norm=2):
        snake_pos = self.get_position(0)
        return self.get_distance_to_target(snake_pos, [snake_pos[0], Y_GRID])

    def get_distance_to_east_wall(self, norm=2):
        snake_pos = self.get_position(0)
        return self.get_distance_to_target(snake_pos, [X_GRID, snake_pos[1]])

    def get_distance_to_west_wall(self, norm=2):
        snake_pos = self.get_position(0)
        return self.get_distance_to_target(snake_pos, [0, snake_pos[1]])

    def is_going_up(self):
        return self.speed[1] < 0

    def is_going_down(self):
        return self.speed[1] > 0

    def is_going_right(self):
        return self.speed[0] > 0

    def is_going_left(self):
        return self.speed[0] < 0
