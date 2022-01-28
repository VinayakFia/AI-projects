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

n_epochs = 50
n_sims = 50
learning_rate = 2

#ML
device = torch.device("cpu")
X = torch.tensor(6)

#plotting

x_plot = []
y_plot = []

for j in range(n_epochs):
    if j >= 1:
        #duplicating first half
        new_thetas = thetas[0:25, :].clone().detach()
        thetas = new_thetas.clone().detach()
        thetas = torch.tensor(torch.vstack((thetas, thetas)))

        #removing first column
        temp_thetas = thetas[:, 1:].clone().detach()

        torch.save(thetas, 'thetas')

        # plotting the points
        x_plot.append(j)
        y_plot.append(average_score)
        
        # naming the x axis
        plt.xlabel('epoch')
        # naming the y axis
        plt.ylabel('mean loss')
        
        # giving a title to my graph
        plt.title('Wee!')
        
        


    i = 0
    for i in range(n_sims):
        if j == 0:
            theta = torch.randn(6)
        else:
            theta = temp_thetas[i-1]
            random_additive = torch.randn(6)/learning_rate
            print(theta)
            print(random_additive)
            theta = torch.add(random_additive, theta)
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
        random_var = random.randint(1, 10)
        ball.dx = 1*random_var
        random_var = random.randint(-10, 10)
        ball.dy = 1*random_var

        pen = turtle.Turtle()
        pen.speed(0)
        pen.color("white")
        pen.penup()
        pen.hideturtle()
        pen.goto(0, 260)

        def paddle_a_up():
            y = paddle_a.ycor()
            if y < 300:
                y += 30
                paddle_a.sety(y)

        def paddle_a_down():
            y = paddle_a.ycor()
            if y > -300:
                y -= 30
                paddle_a.sety(y)

        def paddle_b_up():
            y = paddle_b.ycor()
            if y < 300:
                y += 30
                paddle_b.sety(y)

        def paddle_b_down():
            y = paddle_b.ycor()
            if y > -300:
                y -= 30
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

        #essentially the loss function
        def ball_dist():
            if ball.xcor()>=0:
                diff = (ball.ycor()-paddle_b.ycor())
                diff_square = diff*diff
                return diff_square
            if ball.xcor()<0:
                diff = (ball.ycor()-paddle_a.ycor())
                diff_square = diff*diff
                return diff_square

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
            #if (ball.xcor() < -340 and ball.xcor() > -350) and (paddle_a.ycor() + 50 > ball.ycor() > paddle_a.ycor() - 50):
                #score_a += 1
                #pen.clear()
                #pen.write("Player A: {} Player B: {}".format(score_a, score_b), align="center", font=("Courier", 20, "normal"))
            #if ball.xcor() > 380:
                #score_a = 0
                #pen.clear()
                #pen.write("Player A: {} Player B: {}".format(score_a, score_b), align="center", font=("Courier", 20, "normal"))
                #ball.goto(0, 0)
                #ball.dx *= -1


            #if (ball.xcor() > 340 and ball.xcor() < 350) and (paddle_b.ycor() + 50 > ball.ycor() > paddle_b.ycor() - 50):
                #score_b += 1
                #pen.clear()
                #pen.write("Player A: {} Player B: {}".format(score_a, score_b), align="center", font=("Courier", 20, "normal"))
            #if ball.xcor() < -380:
                #score_b = 0
                #pen.clear()
                #pen.write("Player A: {} Player B: {}".format(score_a, score_b), align="center", font=("Courier", 20, "normal"))
                #ball.goto(0, 0)
                #ball.dx *= -1


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
            
            if time_difference > 0.000005:
                toc = tic

                #running the ai
                AI_predict(X, theta)

                #running the game frame
                [score_a, score_b] = main(score_a, score_b)

                #forwading a frame (same method as stablising framerate) [PREVIOUS METHOD]
                #num_frames +=1
                #frame_diff = num_frames-num_current_frames

                #stop after 36 frames such that the ball has reached the x point of the paddle
                if ball.xcor()>=340:
                    true_var = 0

                    #loss and conversion to tensors
                    loss_tensor = torch.tensor([ball_dist()])
                    row = torch.hstack((loss_tensor, theta))

                    #checking if first through and stacking
                    if i==0:
                        thetas = torch.tensor(row)
                    else:
                        thetas = torch.tensor(torch.vstack((row, thetas)))

                        #sorting algorithim
                        k=0
                        for k in range (i):
                            if thetas[(k-1), 0] >= thetas[(k), 0]:
                                x = thetas.clone().detach()
                                thetas[(k-1)] = x[(k)]
                                thetas[(k)] = x[(k-1)]
                            else:
                                k = n_sims
                        printing = thetas[:, 0]
                        #print(printing)

                    #debuging
                    m = thetas.size()
                    #print(m)
                    turtle.clearscreen()

                    #x = thetas.clone().detach()
                    average_score = 0

                    if i>0:
                        average_score = torch.tensor(torch.mean(thetas, 0))
                        average_score = average_score[0].item()

                    pen.clear()
                    pen.write("Sim: {} Epoch: {} Mean: {}".format(i+1, j+1, average_score), align="center", font=("Courier", 20, "normal"))

# function to show the plot
plt.plot(x_plot, y_plot)
plt.show()
print(thetas)