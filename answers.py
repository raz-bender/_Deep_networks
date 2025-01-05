r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers

part1_q1 = r"""
**Your answer:**
1) **False** , The test set is a set of elements used to evaluate our model's performance on data it has not been 
trained on, simulating real-world unseen data. The in-sample loss (training error) depends only on the instances 
in the training set — the data that the model has seen. For example, we could have an overfitted model that 
performs poorly on the test set but has a low in-sample error, or we could have a well-generalized model that 
performs well on both the training and test sets, showing low in-sample and test error.

2) **False** , in the case that we will take a single data element into the training set , the module will not be able 
to make any meaningful prediction , with only one example the module cannot distinguish between classes that neither 
one is in the training set , nor can it generalize the label it has seen. In conclusion, the data split can affect the 
model's results. 

3) **True** , Cross-validation is done using the validation set, which is split from the training dataset.
Its main goal is to evaluate how well the chosen model performs while training. However, the test set should remain disjointed 
and must not be used during cross-validation or training to ensure that the evaluation of the model on unseen data 
is correctly simulated.

4) **True** , The generalization error measure how well the model performs on unseen data that the model was not trained
on , and represents the model's ability to generalize from the training data.
After performing cross-validation, the validation-set performance in each fold serves as a test set that checks the 
model's performance on data it was not trained on , and attempts to approximate a test he model’s generalization error.  
"""

part1_q2 = r"""
**Your answer:**
The current friends model is overfitted to the training set , in order to combat this , he decided to add an 
regularization term $\lambda$ to the loss , and is attempting to find a $\lambda$ that will minimize the models
generalization error.
The friend's approach might be considered justified , because the added term will  combat overfitting by penalizing 
large weights (Occam's razor), and by training the model with different $\lambda$ and selecting the best 
preforming one , we are able to achieve a better result as we otherwise would , if $\lambda$ was chosen at random.

But , the approach that goes over an finite amount of $\lambda$ is not likely to find an optimal solution of 
$\lambda \in \mathbb{R}$ , furthermore by iterating $\lambda$ over and over on the same test-set, we might 
come back to the same problem where $\lambda$ is overfitted to the test-set and thus , the test-set in a way is part of 
the training set , and does not help us minimize the generalization error .


"""

# ==============
# Part 2 answers
part2_q1 = r"""
**Your answer:**
$$
\max\left(0, \Delta+ \vectr{w_j} \vec{x_i} - \vectr{w_{y_i}} \vec{x_i}\right)
$$
In the case of $\Delta > 0$ : $\Delta$ represents the certainty that we demand from the prediction , as 
$\Delta$ increases we demand that the difference between the predicted class score and the other classes score will be
larger , in order to not get penalized. 


In the case of $\Delta < 0 $ : $\Delta$ represents the tolerance for wrong predictions , as $\Delta$ decreases we allow 
the modal to be less certain in the classes prediction. That is because when $\Delta < 0$ we are not penalizing when the wrong 
prediction score $\vectr{w_j} \vec{x_i}$ is greater than the correct class score $\vectr{w_{y_i}} \vec{x_i}$ by a margin 
of $|\Delta|$ or less, even though the classification is wrong.
"""


part2_q2 = r"""
**Your answer:**

Given the images of the weights in the visualization section we can understand that the linear model is learning the most
 significant pixels for each classification. At some of the images we can actually see the number shines in brighter color 
 out from the image. These bright colors are equivalent to a higher value in the weights tensor, which means that the model
 pays more attention to these pixels when trying to classify to the given image class.
 
 In the test-set examples printing, in the third row we can see a 4 classified as a 6. When looking at the weights image
 matching the 4 we can see a focus on the middle pixels of the input image. But on this sample the image is dark in the middle 
 and a bit lower, so the score was too low for it to be classified as 4. Another example is the 2 classified as 7 in the last 
 column. We can see that the 2 is sitting a bit lower than the place the the weights expect it to be, which causing the wrong classification. 

"""

part2_q3 = r"""
**Your answer:**

Based on the training set graph , we may conclude that our learning rate is good.


In the first case, where the learning rate's hyper-parameter is too low, our graph would have converged slower,
and for the same amount of epochs the graph may not fully converge. 
In the corresponding graph, this would appear as a gradual or shallow slope, indicating progress,
 but the model still hasn't reached an optimal solution by the end of the specified epochs.
 
In the second case, where the learning rate's hyper-parameter is too high, it would have resulted in a more steep 
learning rate (slope) ,causing the loss to decrease too quickly. The graph would have probably converged to a higher loss value.


Based on the graph of the training of the test set accuracy, we would say that our model is slightly overfitted to
the training set, because there is a slight margin between the model's accuracy on the train and valid set.
"""

# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
**Your answer:**

The ideal pattern in the residual plot is a straight horizontal line along the axis ($y-\hat{y}=0$). 
Based on the residual plots and the mse score for each plot, we can conclude that the cross validation approach works 
better than the top 5 features approach, as it's mse score is lower.   
"""

part3_q2 = r"""
**Your answer:**


1. The model is still a linear model. Even by adding a non-linear feature, we are still
searching for a linear solution on an expanded feature space that includes a new dimension. 
The new dimension represents the transformed, non-linear feature, but the model’s decision boundary remains linear in the transformed space.


2. Let $f(\bar{x})$ be the non-linear function on the feature space that underlines the true relationship between 
the input and target features. By adding a new non-linear 
feature $f(\bar{x})$, the model can  represent this relationship explicitly and fit the data more accurately,
thus allowing it to use the true underlying non-linear relationships of the data.$\; \; \; \; \; \;$   
But, for the added non-linear feature to improve the model's fittness, it must be able to capture and predict 
some of the patterns in the underlying feature relationships.


3. The decision boundary of the new features would still be an hyperplane not in the original features space but in the transformed
 space. By adding new features that are non-linear
with regards to the original features, we are creating a new hyperspace with additional dimensions,
 that are linear on the new features, and thus the decision boundary remains linear on the new feature space. 
"""

part3_q3 = r"""
**Your answer:**
1. By using np.logspace, we define a set of points such that each pair of adjacent points is equidistant in logarithmic
 space. This means that the distribution of points will be denser for lower values of $\lambda$ 
 and more spread out for higher values when considered on a linear scale. As a result, 
 np.logspace generates points that span different orders of magnitude, providing a more 
 uniform coverage across a wide range of values on a logarithmic scale.

Thus by using np.logspace we are opting to measure the scale of $\lambda$ rather
than to find the optimal specific value ,as by using np.linspace, over the entire range, which would result in an 
exponential increase to our computation time without necessarily adding a meaningful benefit.

2. The modal was fitted $|degree_range| \cdot |lambda_range| \cdot k = 3 \cdot 20 \cdot 3 = 180$  times.
Using cross-validation (CV), the model was fitted $k$ times for each combination of degree and $\lambda$ .

Thus by applying CV over all combinations ($|degree_range| \cdot |lambda_range|$) the modal was fitted a total of 
$|degree_range| \cdot |lambda_range| \cdot k$ times 
"""

# ==============

# ==============
