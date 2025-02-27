Download link :https://programming.engineering/product/linear-regression-assignment-2-of-the-machine-learning-1-2/

# Linear-Regression-Assignment-2-of-the-Machine-Learning-1
Linear Regression Assignment 2 of the Machine Learning 1
1 Introduction

Linear regression is an algorithm used to visualize a relation-ship between two variables, a dependent variable, and an inde-pendent variable.

The independent variable stands by itself and isn’t impacted by the other variable. While the value of the dependent variable will fluctuate according to the independent variable. So in a regression problem, we try to find the relationship between the independent and the dependent variable, in order to be able to approximate the target variable.

It is a linear problem so we use a first-order polyno-mial function to approximate the data. Indeed, for a one-dimensional problem, the approximation will be a line, for a two-dimensional problem, a plane, and so on.

2 Theory of Linear Regression

2.1 One-dimensional problem

The linear regression problem can be formalized in the follow-ing way:

define a linear function such that, given an observation (x), computes the target (t) that best approximates the real value (y).

t = wx, t ≈ y

(1)

Linear regression aims to compute the best value of the param-eter w that better approximates each observation, to do so it must minimize the mean error.

We use a square error for w calculations because it grows more than linearly, providing heavier weights to larger errors, it is even and differentiable:

2.2 Adding the intercept

It is possible to improve our regression model by adding the intercept factor, w0, to equation 1:

t = w1x + w0

(5)

The two parameters, w0 and w1 can then be computed as fol-lows:

N

(xl − x¯) tl

¯

w1 =

l=1

− t

(6)

N

2

l=1 (xl − x¯)

¯

(7)

w0 = t − w1x¯

2.3 Multi-dimensional linear regression

Now that we have the basics, we can generalize to the multi-dimensional problem.

X now is a matrix of observations, and w is a vector of the approximation parameters:

X =

x2

w = w1

(8)

x1

w0

…

w

n

xn

…

We obtain the final output (y) of the linear regression as a ma-trix product:

y =

1

x2,1

…

x2,n

w1

= Xw

(9)

1

x1,1

…

x1,n

w0

1

x

…

x

w

n

n,1

n,n

… … … …

…


3 Assignment

The assignment is composed of three tasks:

Get data;

Fit a linear regression model;

Test regression model.

3.1 Get data

For this assignment we will use two data sets, provided in a csv format:

the Turkish stock exchange data (turkish-se-SP500vsMSCI);

the 1974 Motor Trend Car Road Tests (mtcarsdata-4features).

The first one has 536 observations and we will use it for one-dimensional problems, while the second has 32 observations and has 4 variables, so we can solve multi-variable problems.

First thing first, we import the data using the read csv() Python function from the Pandas library, in order convert them into dataframes.

We don’t have to do particular data pre-processing since the data sets are already ready to use.

3.2 Fit a linear regression model

Once we imported our data we have to compute the linear re-gression parameters in four different cases:

One-dimensional problem without intercept on the Turk-ish stock exchange data;

Compare graphically the solution obtained on different random subsets (10%) of the whole data set;

One-dimensional problem with intercept on the Motor Trends car data, using columns mpg and weight;

Multi-dimensional problem on the Motor Trends car data, using all four columns (predict mpg with the other three columns).

Before starting we prepare in advance two functions to print the linear regression graphs, one for the one-dimensional problem and one for the multi-dimensional problem. The functions use the matplotlib Python’s library.

Take notice that in the code y is used to indicate the target t.

3.2.1 1

In order to calculate the linear regression (without intercept) we create a function, linear regression one dim no intercept(), which takes as arguments a dataframe, the column of the ob-servation variable, and the column of the target variable. The function calculates the parameter w, which is called slope in the code, then returns a lambda function that corresponds to the linear regression function. The result is shown in figure 1.

3.2.2

2

Now

we

create

4 random

subsets

of

10% of

the

data

set.

Then

we pass

directly

the

function

lin-

ear regression one dim no intercept() as

an

argument

in

a function that creates a plot with all the 4 linear regression lines and their relative subsets used for computing the lines. The result can be analyzed in figure 2.

3.2.3 3

For this task, we create another function, called lin-ear regression one dim(), that calculates the linear regression line taking into account the intercept. We pass the weight col-umn as an independent variable and the mpg as a target vector. The calculations are the ones shown in subsection 2.2. The plot of this task is in figure 3.

3.2.4 4

In this last step, we create a function, called lin-ear regression multidim(), in order to predict mpg using the other three columns on the Motor Trends car data set. This time we pass a matrix of observations as an argument instead of a vector.

According to the theory stated in subsection 2.3, we compute the Moore-Penrose pseudoinverse, then use it to compute the w factor, and finally return a lambda for the linear regres-sion function, similarly to the previous functions. Note that in Python we use the symbol @ to perform matrix multiplications. Since it’s a multi-dimensional problem we don’t plot anything but instead, we print the predicted output on the console (figure 4).

3.3 Test regression model

The last task requires running again points 1, 3, and 4 from the previous task using 5% of the data, computing the MSE on the training data, and repeating the process for the remaining 95% of the data. Then repeat for different training-test random splits.

Here the only new thing is that I introduced two functions, one to compute the MSE in a one-dimensional problem and one to calculate the MSE for a multi-dimensional problem.

Lastly, to compare the performances for different training-test sets, I plotted a histogram for the MSE using the function mse histograms().

Figures 5, 6, 7 show the linear regression problem for the differ-ent cases with the 5%-95% split. While 8 show the respective Mean Square Errors.

Note that for the second data set, if we take only 5% of the data, it means we have only 2 observation points, so the regression line will fit the data perfectly. We can also prove it by observ-ing that the MSE (third line of figure 8) is really small.

Finally, we compute the MSE for 100 random splits and plot the histograms (figure 9).


4 Results


Figure 1. One-dim linear regression without intercept.
