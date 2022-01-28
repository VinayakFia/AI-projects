from ast import Return
import turtle
import time
from numpy import array
import numpy as np
import torch
import torch.optim as optim
import random
import matplotlib.pyplot as plt

turtle.speed(0)

#ML
device = torch.device("cpu")
X = torch.tensor(6)

#theta
theta = torch.tensor([1.6060e-01, -9.5109e+00, -1.6922e-01, -3.5562e-01, 1.5282e+00,  5.3522e-01])

#print(theta)
#setup
num_frames = 0
num_current_frames = 0

score_a = 0
score_b = 0

screen = turtle.Screen()
screen.title("Pong")
screen.bgcolor("black")
screen.setup(width=800, height=600)
screen.tracer(0)

paddle_a = turtle.Turtle()
paddle_a.speed(0)
paddle_a.shape("square")
paddle_a.color("white")
paddle_a.shapesize(stretch_wid=5, stretch_len=1)
paddle_a.penup()
paddle_a.goto(-350, 0)

paddle_b = turtle.Turtle()
paddle_b.speed(0)
paddle_b.shape("square")
paddle_b.color("white")
paddle_b.shapesize(stretch_wid=5, stretch_len=1)
paddle_b.penup()
paddle_b.goto(350, 0)

ball = turtle.Turtle()
ball.speed(0)
ball.shape("circle")
ball.color("white")
ball.penup()
ball.goto(0, 0)
random_var = random.randint(1, 8)
ball.dx = 1*random_var
random_var = random.randint(-10, 10)
ball.dy = 1*random_var

pen = turtle.Turtle()
pen.speed(0)
pen.color("white")
pen.penup()
pen.hideturtle()
pen.goto(0, 260)

mov_speed = 5

def paddle_a_up():
    y = paddle_a.ycor()
    if y < 300:
        y += mov_speed
        paddle_a.sety(y)

def paddle_a_down():
    y = paddle_a.ycor()
    if y > -300:
        y -= mov_speed
        paddle_a.sety(y)

def paddle_b_up():
    y = paddle_b.ycor()
    if y < 300:
        y += mov_speed
        paddle_b.sety(y)

def paddle_b_down():
    y = paddle_b.ycor()
    if y > -300:
        y -= mov_speed
        paddle_b.sety(y)

screen.listen()
screen.onkeypress(paddle_a_up, "w")
screen.onkeypress(paddle_a_down, "s")
screen.onkeypress(paddle_b_up, "Up")
screen.onkeypress(paddle_b_down, "Down")

tic = 0
toc = 0

#Previous Tings
prev_xcor = 0
prev_ycor = 0


def AI_predict(X, theta):
    if ball.xcor()>=0:
        X = torch.tensor(([0, (paddle_b.ycor()-ball.ycor())/300, ball.xcor()/400, ball.ycor()/300, prev_xcor/400, prev_ycor/300]))
        prediction = torch.sum(torch.mul(X, theta))  
        if prediction >= 0:
            paddle_b_up()
        else:
            paddle_b_down()
        #print(prediction)
    if ball.xcor()<0:
        X = torch.tensor(([0, (paddle_a.ycor()-ball.ycor())/300, (ball.xcor()/400)*-1, ball.ycor()/300, (prev_xcor/400)*-1, prev_ycor/300]))
        prediction = torch.sum(torch.mul(X, theta))
        if prediction >= 0:
            paddle_a_up()
        else:
            paddle_a_down()
        #print(prediction)

def main(score_a, score_b):
    screen.update()

    # move the ball
    ball.setx(ball.xcor() + ball.dx)
    ball.sety(ball.ycor() + ball.dy)

    # border checking
    if ball.ycor() > 280:
        ball.sety(280)
        ball.dy *= -1

    if ball.ycor() < -280:
        ball.sety(-280)
        ball.dy *= -1

    #left and right
    if (ball.xcor() < -340 and ball.xcor() > -350) and (paddle_a.ycor() + 50 > ball.ycor() > paddle_a.ycor() - 50):
        score_a += 1
        pen.clear()
        pen.write("Player A: {} Player B: {}".format(score_a, score_b), align="center", font=("Courier", 20, "normal"))
    if ball.xcor() > 380:
        score_a = 0
        pen.clear()
        pen.write("Player A: {} Player B: {}".format(score_a, score_b), align="center", font=("Courier", 20, "normal"))
        ball.goto(0, 0)
        ball.dx *= -1

    if (ball.xcor() > 340 and ball.xcor() < 350) and (paddle_b.ycor() + 50 > ball.ycor() > paddle_b.ycor() - 50):
        score_b += 1
        pen.clear()
        pen.write("Player A: {} Player B: {}".format(score_a, score_b), align="center", font=("Courier", 20, "normal"))
    if ball.xcor() < -380:
        score_b = 0
        pen.clear()
        pen.write("Player A: {} Player B: {}".format(score_a, score_b), align="center", font=("Courier", 20, "normal"))
        ball.goto(0, 0)
        ball.dx *= -1

    # paddle and ball collisions
    if (ball.xcor() > 340 and ball.xcor() < 350) and (paddle_b.ycor() + 50 > ball.ycor() > paddle_b.ycor() - 50):
        ball.setx(340)
        ball.dx *= -1

    if (ball.xcor() < -340 and ball.xcor() > -350) and (paddle_a.ycor() + 50 > ball.ycor() > paddle_a.ycor() - 50):
        ball.setx(-340)
        ball.dx *= -1

    return array([score_a, score_b])



true_var = 1

#game loop
while true_var == 1:
    #to create a stable framerate
    tic = time.perf_counter()
    time_difference = tic-toc
    
    if time_difference > 0.005:
        toc = tic

        #running the ai
        AI_predict(X, theta)

        [score_a, score_b] = main(score_a, score_b)