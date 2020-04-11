---
title: Bayesian Optimization
date: 2020-04-11 10:59:48
tags: 
- AutoML
- Optimization
---



According to Wikipedia, Bayesian optimization (BO) is a sequential design strategy for global optimization of black-box functions that do not require derivatives.

# Study Objective

In this post, we will:

1. Review optimization algorithms. Compare Bayesian optimization with gradient descent.

2. Understand the process of Bayesian optimization.

3. Write a simple Bayesian optimization algorithm from scratch.

4. Use BO not from scratch.

# Optimization

Optimization is to find the global max or min from the function. Besides BO, we also use grid search, random search and gradient descent.  The first two can be used on any function if computation power is not the problem. The third is used only if the function is convex and derivative. Bayesian optimization has no such limitations. However, the function is supposed to be smooth and continuous, otherwise, I would recommend grid search and random search.

# Process

I think the whole process can be demonstrated as the following trinity:

![Bayesian Optimization Trinity](/images/automl/bayesian_optimization_process.png)

Here the **objective function** is the black function that we optimize. It produces samples (x, y). Usually, it is very costly to run the objective function, such that we introduce a **surrogate function** to predict the objective function. The prediction is mean and standard deviation of the objective function, which are used by the **acquisition function** in searching for the next x to explore or exploit. The next x is then consumed by the objective function. The loop goes on until the max or min is found. 

The surrogate function is usually the Gaussian process regressor while the acquisition function has many options. They are:

1. Probability     of Improvement (PI).
2. Expected     Improvement (EI).
3. Upper/Lower     Confidence Bound (LCB/UCB).

Here we only focus on what we use, the UCB, the most straightforward one. If we ignore weight, UCB can be written as:

𝑈𝐶𝐵(𝑥)=𝜇(𝑥)+𝜎(𝑥)

𝑥=argmax 𝑈𝐶𝐵(𝑥)

 

We do all of this inside the **search space**.

The whole process:

1. Initiate     x with the min and max of the search space.
2. Calculate     y using x and objective function.
3. Fit     the surrogate function.
4. Find     new x using the acquisition function.
5. Go     to 1

 

# Python Code

Let’s first define and plot the objective function as following:

```python
def objective(x):
    return ((x-0.47)**2 * math.sin(3 * x))
```



 

![Objective](/images/automl/objective.png)

Here the maximum (x=0.87, y= 0.0811051) is also plotted as a blue point.

 

Now, this function is non-convex, thus gradient descent cannot be used. We use BO. The surrogate function is the Gaussian process regressor and UCB is used as the acquisition function.

```python
#uppper confidence bound (UCB)
#beta = 1
def acquisition(mean, std):
    mean=mean.flatten()

    #UCB
    upper=mean+std
    #argmax
    max_at=np.argmax(upper)
    return X[max_at]

#surrogate
surrogate = GaussianProcessRegressor()
```



 

Code for the whole process

```python
#step 0 Initiate x with the min and max of the search space.
xsamples=np.array([[0],[1]])

#step 1 Calculate y using x and objective function.
ysamples=np.array([objective(x) for x in xsamples])

for i in range(4):
    #step 2 Fit the surrogate function.
    surrogate.fit(xsamples, ysamples)
    mean, std=surrogate.predict(X, return_std=True)
    
    #step 3 Find new x using acquisition function.
    new_x=acquisition(mean, std)
    
    #step 4 Go to 1
    new_y=objective(new_x)

    #plot
    plot(X, y, xsamples, ysamples, mean, std, new_x, new_y, i)
    xsamples=np.vstack((xsamples, new_x))
    ysamples=np.vstack((ysamples, new_y))
```



Step 0:

In step 0, we simply produce two sample x=0 and x=1. 

Iteration 0:

In iteration 0, the two samples are used by the surrogate function to generate mean and std(green). The acquisition function finds the max UCB when x=0.53(red). Objective(0.53) is 0.00359934. Now a new point (0.53, 0.00359934) is found. We give the three points to iteration 1.

![Bayesian Optimization Iteration 0](/images/automl/bayesian_optimization_iteration_0.png)

Iteration 1:

![Bayesian Optimization Iteration 1](/images/automl/bayesian_optimization_iteration_1.png)

Iteration 2:

![Bayesian Optimization Iteration 2](/images/automl/bayesian_optimization_iteration_2.png)

Iteration 3:

![Bayesian Optimization Iteration 3](/images/automl/bayesian_optimization_iteration_3.png)

The new point(red) here is (0.53, 0.00359934) and it is just the max. We can see that BO can find the optimum in just 4 iterations. And we call the objective function only 6 times. If we use the grid search, that will be 100 times. 

The complete version of the code can be found here:

https://github.com/EricWebsmith/machine_learning_from_scrach/blob/master/bayesian_optimization.ipynb

# Bayesian Optimization not from scratch

There are many tools for BO. One of them is Hyperopt. The following is just a simple demonstration of that.

```python
from hyperopt import fmin, tpe, hp
best = fmin(
    fn=lambda x:-objective(x),
    space=hp.uniform('x', 0, 1),
    algo=tpe.suggest,
    max_evals=100)
print(best) 
```

# Reference

https://en.wikipedia.org/wiki/Bayesian_optimization

 

 

 

 