import pygame
import time
import ui
import random


def play(screen, snake, apple, walls):
    myfont = pygame.font.SysFont('Comic Sans MS', 30)
    score = 0

    while not snake.dead:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                snake.dead = True

        if random.randint(0, 1):
            if snake.speed[0] == 0:
                key = random.choice([pygame.K_LEFT, pygame.K_RIGHT])
            else:
                key = random.choice([pygame.K_UP, pygame.K_DOWN])
            snake.change_direction(key)

        if snake.eat(apple):
            snake.grow()
            apple.new_random()
            score += 1

        snake.update(walls)
        textsurface = myfont.render('Score: {}'.format(score), False, ui.WHITE)

        # Draw Everything
        screen.fill(ui.BLACK)
        screen.blit(textsurface, (ui.WIDTH + 50, 50))
        snake.draw(screen)
        apple.draw(screen)
        walls.draw(screen)
        pygame.display.flip()

        time.sleep(150.0 / 1000.0)

    pygame.quit()
    return score
