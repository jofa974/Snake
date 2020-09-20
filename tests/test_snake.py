from components.apple import Apple
from components.snake import Snake
from ui import BASE_SPEED

apple = Apple()
apple.new(20, 20)
apple_pos = apple.get_position()


def test_get_distance_to_target():
    snake = Snake(20, 40, (0, -BASE_SPEED))
    dist = snake.get_distance_to_target(snake.get_position(0), apple_pos, norm=1)
    assert dist == 20

    snake = Snake(0, 20, (0, BASE_SPEED))
    dist = snake.get_distance_to_target(snake.get_position(0), apple_pos, norm=1)
    assert dist == 20

    snake = Snake(20, 20, (0, BASE_SPEED))
    dist = snake.get_distance_to_target(snake.get_position(0), apple_pos, norm=1)
    assert dist == 0

    snake = Snake(0, 0, (0, BASE_SPEED))
    dist = snake.get_distance_to_target(snake.get_position(0), apple_pos, norm=1)
    assert dist == 40


def test_get_next_pos_left():
    snake = Snake(20, 20, (BASE_SPEED, 0))
    next_left = snake.get_next_pos_left()
    assert next_left == (20, 19)

    snake = Snake(20, 20, (-BASE_SPEED, 0))
    next_left = snake.get_next_pos_left()
    assert next_left == (20, 21)

    snake = Snake(20, 20, (0, BASE_SPEED))
    next_left = snake.get_next_pos_left()
    assert next_left == (21, 20)

    snake = Snake(20, 20, (0, -BASE_SPEED))
    next_left = snake.get_next_pos_left()
    assert next_left == (19, 20)


def test_get_next_pos_right():
    snake = Snake(20, 20, (BASE_SPEED, 0))
    next_left = snake.get_next_pos_right()
    assert next_left == (20, 21)

    snake = Snake(20, 20, (-BASE_SPEED, 0))
    next_left = snake.get_next_pos_right()
    assert next_left == (20, 19)

    snake = Snake(20, 20, (0, BASE_SPEED))
    next_left = snake.get_next_pos_right()
    assert next_left == (19, 20)

    snake = Snake(20, 20, (0, -BASE_SPEED))
    next_left = snake.get_next_pos_right()
    assert next_left == (21, 20)


def test_is_food_ahead():
    # Snake going up
    snake = Snake(0, 40, (0, -BASE_SPEED))
    assert snake.is_food_ahead(apple_pos)
    # Snake going down
    snake = Snake(0, 0, (0, BASE_SPEED))
    assert snake.is_food_ahead(apple_pos)
    # Snake going right
    snake = Snake(0, 20, (BASE_SPEED, 0))
    assert snake.is_food_ahead(apple_pos)
    # Snake going left
    snake = Snake(40, 20, (-BASE_SPEED, 0))
    assert snake.is_food_ahead(apple_pos)


def test_is_food_right():
    # Snake going down
    snake = Snake(30, 0, (0, BASE_SPEED))
    assert snake.is_food_right(apple_pos)
    snake = Snake(0, 0, (0, BASE_SPEED))
    assert not snake.is_food_right(apple_pos)

    # Snake going up
    snake = Snake(0, 30, (0, -BASE_SPEED))
    assert snake.is_food_right(apple_pos)
    snake = Snake(30, 30, (0, -BASE_SPEED))
    assert not snake.is_food_right(apple_pos)

    # Snake going right
    snake = Snake(0, 0, (BASE_SPEED, 0))
    assert snake.is_food_right(apple_pos)
    snake = Snake(0, 30, (BASE_SPEED, 0))
    assert not snake.is_food_right(apple_pos)

    # Snake going left
    snake = Snake(30, 30, (-BASE_SPEED, 0))
    assert snake.is_food_right(apple_pos)
    snake = Snake(30, 0, (-BASE_SPEED, 0))
    assert not snake.is_food_right(apple_pos)


def test_is_food_left():
    # Snake going down
    snake = Snake(0, 0, (0, BASE_SPEED))
    assert snake.is_food_left(apple_pos)
    snake = Snake(30, 0, (0, BASE_SPEED))
    assert not snake.is_food_left(apple_pos)

    # Snake going up
    snake = Snake(30, 30, (0, -BASE_SPEED))
    assert snake.is_food_left(apple_pos)
    snake = Snake(0, 30, (0, -BASE_SPEED))
    assert not snake.is_food_left(apple_pos)

    # Snake going right
    snake = Snake(0, 30, (BASE_SPEED, 0))
    assert snake.is_food_left(apple_pos)
    snake = Snake(0, 0, (BASE_SPEED, 0))
    assert not snake.is_food_left(apple_pos)

    # Snake going left
    snake = Snake(30, 0, (-BASE_SPEED, 0))
    assert snake.is_food_left(apple_pos)
    snake = Snake(30, 30, (-BASE_SPEED, 0))
    assert not snake.is_food_left(apple_pos)
