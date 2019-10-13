from components.apple import Apple
from components.snake import Snake

BASE_SPEED = 20
apple = Apple()
apple.new(20, 20)


def test_get_distance_to_apple():
    snake = Snake(20, 40, (0, -BASE_SPEED))
    dist = snake.get_distance_to_apple(snake.get_position(0), apple, norm=1)
    assert dist == 20

    snake = Snake(0, 20, (0, BASE_SPEED))
    dist = snake.get_distance_to_apple(snake.get_position(0), apple, norm=1)
    assert dist == 20

    snake = Snake(20, 20, (0, BASE_SPEED))
    dist = snake.get_distance_to_apple(snake.get_position(0), apple, norm=1)
    assert dist == 0

    snake = Snake(0, 0, (0, BASE_SPEED))
    dist = snake.get_distance_to_apple(snake.get_position(0), apple, norm=1)
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
    snake = Snake(0, 40, (0, -BASE_SPEED))
    assert snake.is_food_ahead(apple)
    snake = Snake(0, 0, (0, BASE_SPEED))
    assert snake.is_food_ahead(apple)

    snake = Snake(0, 20, (BASE_SPEED, 0))
    assert snake.is_food_ahead(apple)
    snake = Snake(40, 20, (-BASE_SPEED, 0))
    assert snake.is_food_ahead(apple)


def test_is_food_right():
    snake = Snake(20, 0, (BASE_SPEED, 0))
    assert snake.is_food_right(apple)
    snake = Snake(10, 0, (0, BASE_SPEED))
    assert snake.is_food_ahead(apple)

    snake = Snake(0, 20, (BASE_SPEED, 0))
    assert snake.is_food_ahead(apple)
    snake = Snake(40, 20, (-BASE_SPEED, 0))
    assert snake.is_food_ahead(apple)
