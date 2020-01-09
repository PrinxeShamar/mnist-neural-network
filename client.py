import pygame
import math
import numpy as np
import tensorflow as tf
import cv2

pygame.init()

number_guesser_model = tf.keras.models.load_model("number_guesser_2.model")


def text_objects(text, font):
    textSurface = font.render(text, True, (0,0,0))
    return textSurface, textSurface.get_rect()


def message_display(window, text, x, y, size):
    largeText = pygame.font.Font('freesansbold.ttf',size)
    TextSurf, TextRect = text_objects(text, largeText)
    TextRect.center = ((x),(y))
    window.blit(TextSurf, TextRect)


class App():
    def __init__(self):
        self.ROWS = 28
        self.COLS = 28
        self.EXTRA = 300

        self.WIDTH = self.ROWS * 28
        self.HEIGHT = self.COLS * 28
        self.NAME = 'Number Guesser'

        self.window = pygame.display.set_mode((self.WIDTH + self.EXTRA, self.HEIGHT))
        pygame.display.set_caption(self.NAME)

        self.drawing = np.zeros((self.ROWS, self.COLS)) * 0.0
        self.drawing_array = np.zeros((1, self.ROWS, self.COLS)) * 0.0

        self.flag = True
        self.loop()

    def draw(self):

        for i in range(0, self.COLS + 1):
            pygame.draw.line(self.window, (0, 0, 0), (self.WIDTH / self.COLS * i, 0),
                             (self.WIDTH / self.COLS * i, self.WIDTH))

        for i in range(0, self.ROWS + 1):
            pygame.draw.line(self.window, (0, 0, 0), (0, self.WIDTH / self.ROWS * i),
                             (self.HEIGHT, self.WIDTH / self.ROWS * i))

        for a in range(0, self.ROWS):
            for b in range(0, self.COLS):
                if self.drawing[a][b] > 0:
                    color = 255 - (255 * self.drawing[a][b])
                    x = ((self.WIDTH / self.COLS) * (a + 1)) - (self.WIDTH / self.COLS)
                    y = ((self.HEIGHT / self.ROWS) * (b + 1)) - (self.HEIGHT / self.ROWS)

                    pygame.draw.rect(self.window, (color, color, color), (x, y, self.HEIGHT / self.ROWS, self.WIDTH / self.COLS))

                '''

                if self.drawing_array[0][a][b] > 0:
                    color = 255 - (255 * self.drawing_array[0][a][b])
                    x = ((self.WIDTH / self.COLS) * (a + 1)) - (self.WIDTH / self.COLS)
                    y = ((self.HEIGHT / self.ROWS) * (b + 1)) - (self.HEIGHT / self.ROWS)

                    pygame.draw.rect(self.window, (color, color, color), (x, y, self.HEIGHT / self.ROWS, self.WIDTH / self.COLS))
                    

                '''


    def loop(self):
        while self.flag:
            pygame.time.delay(1)

            events = pygame.event.get()

            for event in events:
                if event.type == pygame.QUIT:
                    self.flag = False
                elif pygame.mouse.get_pressed()[0]:
                    x, y = event.pos
                    row = math.floor(x / self.ROWS)
                    col = math.floor(y / self.COLS)

                    if row < self.ROWS and col < self.COLS:
                        self.drawing[row][col] = 1.0

                        self.drawing[row][col + 1] = 1.0
                        self.drawing[row][col - 1] = 1.0

                        self.drawing[row + 1][col] = 1.0
                        self.drawing[row - 1][col] = 1.0

                        self.drawing[row + 1][col - 1] = 1.0
                        self.drawing[row + 1][col + 1] = 1.0

                        self.drawing[row - 1][col - 1] = 1.0
                        self.drawing[row - 1][col + 1] = 1.0

                        self.drawing_array[0][col][row] = 1.0

                        if self.drawing_array[0][col - 1][row] <= 0:
                            self.drawing_array[0][col - 1][row] = 0.5

                        if self.drawing_array[0][col + 1][row] <= 0:
                            self.drawing_array[0][col + 1][row] = 0.5

                        if self.drawing_array[0][col][row + 1] <= 0:
                            self.drawing_array[0][col][row + 1] = 0.5

                        if  self.drawing_array[0][col][row - 1] <= 0:
                            self.drawing_array[0][col][row - 1] = 0.5

                        if self.drawing_array[0][col + 1][row + 1] <= 0:
                            self.drawing_array[0][col + 1][row + 1] = 0.25

                        if self.drawing_array[0][col - 1][row + 1] <= 0:
                            self.drawing_array[0][col - 1][row + 1] = 0.25

                        if self.drawing_array[0][col + 1][row + 1] <= 0:
                            self.drawing_array[0][col + 1][row - 1] = 0.25

                        if self.drawing_array[0][col - 1][row - 1] <= 0:
                            self.drawing_array[0][col - 1][row - 1] = 0.25

            self.window.fill((255, 255, 255))

            self.draw()

            prediction = number_guesser_model.predict([self.drawing_array])
            if prediction[0][np.argmax(prediction[0])] > 0.5:
                message_display(self.window, str(np.argmax(prediction[0])), self.WIDTH + (self.EXTRA / 2) , 90, 150)

            pygame.display.update()

if __name__ == '__main__':
    app = App()