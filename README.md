# machine-learning-1-week-5-model-selection-solved
**TO GET THIS SOLUTION VISIT:** [Machine Learning 1 Week 5-Model Selection Solved](https://www.ankitcodinghub.com/product/machine-learning-1-week-5-model-selection-solved/)


---

📩 **If you need this solution or have special requests:** **Email:** ankitcoding@gmail.com  
📱 **WhatsApp:** +1 419 877 7882  
📄 **Get a quote instantly using this form:** [Ask Homework Questions](https://www.ankitcodinghub.com/services/ask-homework-questions/)

*We deliver fast, professional, and affordable academic help.*

---

<h2>Description</h2>



<div class="kk-star-ratings kksr-auto kksr-align-center kksr-valign-top" data-payload="{&quot;align&quot;:&quot;center&quot;,&quot;id&quot;:&quot;98754&quot;,&quot;slug&quot;:&quot;default&quot;,&quot;valign&quot;:&quot;top&quot;,&quot;ignore&quot;:&quot;&quot;,&quot;reference&quot;:&quot;auto&quot;,&quot;class&quot;:&quot;&quot;,&quot;count&quot;:&quot;0&quot;,&quot;legendonly&quot;:&quot;&quot;,&quot;readonly&quot;:&quot;&quot;,&quot;score&quot;:&quot;0&quot;,&quot;starsonly&quot;:&quot;&quot;,&quot;best&quot;:&quot;5&quot;,&quot;gap&quot;:&quot;4&quot;,&quot;greet&quot;:&quot;Rate this product&quot;,&quot;legend&quot;:&quot;0\/5 - (0 votes)&quot;,&quot;size&quot;:&quot;24&quot;,&quot;title&quot;:&quot;Machine Learning 1 Week 5-Model Selection Solved&quot;,&quot;width&quot;:&quot;0&quot;,&quot;_legend&quot;:&quot;{score}\/{best} - ({count} {votes})&quot;,&quot;font_factor&quot;:&quot;1.25&quot;}">

<div class="kksr-stars">

<div class="kksr-stars-inactive">
            <div class="kksr-star" data-star="1" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="2" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="3" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="4" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="5" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>

<div class="kksr-stars-active" style="width: 0px;">
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>
</div>


<div class="kksr-legend" style="font-size: 19.2px;">
            <span class="kksr-muted">Rate this product</span>
    </div>
    </div>
<div class="page" title="Page 1">
<div class="layoutArea">
<div class="column">
Exercise 1: Bias and Variance of Mean Estimators

Assume we have an estimator θˆ for a parameter θ. The bias of the estimator θˆ is the difference between the true value for the estimator, and its expected value

Bias(θˆ) = E􏰏θˆ − θ􏰐.

If Bias(θˆ) = 0, then θˆ is called unbiased. The variance of the estimator θˆ is the expected square deviation

</div>
</div>
<div class="layoutArea">
<div class="column">
from its expected value

Var(θˆ) = E􏰏(θˆ − E[θˆ])2􏰐. The mean squared error of the estimator θˆ is

Error(θˆ) = E􏰏(θˆ − θ)2􏰐 = Bias(θˆ)2 + Var(θˆ).

Let X1, . . . , XN be a sample of i.i.d random variables. Assume that Xi has

Calculate the bias, variance and mean squared error of the mean estimator:

1 􏰉N

</div>
<div class="column">
mean μ and

</div>
<div class="column">
variance σ2.

</div>
</div>
<div class="layoutArea">
<div class="column">
μˆ = α · N

Exercise 2: Bias-Variance Decomposition for Classification (30 P)

</div>
</div>
<div class="layoutArea">
<div class="column">
where α is a parameter between 0 and 1.

</div>
</div>
<div class="layoutArea">
<div class="column">
The bias-variance decomposition usually applies to regression data. In this exercise, we would like to obtain similar decomposition for classification, in particular, when the prediction is given as a probability distribution over C classes. Let P = [P1, . . . , PC ] be the ground truth class distribution associated to a particular input pattern. Assume a random estimator of class probabilities Pˆ = [Pˆ1,…,PˆC] for the same input pattern. The error function is given by the expected KL-divergence between the ground truth and the estimated probability distribution:

Error = E􏰏DKL(P ||Pˆ)􏰐 = E􏰏 􏰌Ci=1 Pi log(Pi/Pˆi)􏰐.

First, we would like to determine the mean of of the class distribution estimator Pˆ. We define the mean as the distribution that minimizes its expected KL divergence from the the class distribution estimator, that is, the distribution R that optimizes

min E􏰏DKL(R||Pˆ)􏰐. R

</div>
</div>
<div class="layoutArea">
<div class="column">
(a) Show that the solution to the optimization problem above is given by expE􏰏logPˆi􏰐

</div>
</div>
<div class="layoutArea">
<div class="column">
R=[R1,…,RC] where Ri=􏰌jexpE􏰏logPˆj􏰐 ∀1≤i≤C.

(Hint: To implement the positivity constraint on R, you can reparameterize its components as Ri = exp(Zi),

</div>
</div>
<div class="layoutArea">
<div class="column">
i=1

</div>
</div>
<div class="layoutArea">
<div class="column">
X i

</div>
</div>
<div class="layoutArea">
<div class="column">
and minimize the objective w.r.t. Z.) (b) Prove the bias-variance decomposition

Error(Pˆ) = Bias(Pˆ) + Var(Pˆ) where the error, bias and variance are given by

Error(Pˆ) = E􏰏DKL(P ||Pˆ)􏰐, Bias(Pˆ) = DKL(P ||R), Var(Pˆ) = E􏰏DKL(R||Pˆ)􏰐. (Hint: as a first step, it can be useful to show that E[log Ri − log Pˆi] does not depend on the index i.)

Exercise 3: Programming (50 P)

Download the programming files on ISIS and follow the instructions.

</div>
</div>
</div>
<div class="page" title="Page 2">
<div class="section">
<div class="layoutArea">
<div class="column">
Exercise sheet 5 (programming) [WiSe 2021/22] Machine Learning 1

</div>
</div>
<div class="layoutArea">
<div class="column">
Part 1: The James-Stein Estimator (20 P)

Let x1, …, xN ∈ Rd be independent draws from a multivariate Gaussian distribution with mean vector μ and covariance matrix Σ = σ2I. It can be shown that the maximum-likelihood estimator of the mean parameter μ is the empirical mean given by:

1N

μˆ M L = N ∑ x i

i=1

Maximum-likelihood appears to be a strong estimator. However, it was demonstrated that the following estimator

(d − 2) ⋅ σ2

μˆ J S = ( 1 − N ) μˆ M L

(a shrinked version of the maximum-likelihood estimator towards the origin) has actually a smaller distance from the true mean when d ≥ 3. This however assumes knowledge of the variance of the distribution for which the mean is estimated. This estimator is called the James-Stein estimator. While the proof is a bit involved, this fact can be easily demonstrated empirically through simulation. This is the object of this exercise.

The code below draws ten 50-dimensional points from a normal distribution with mean vector μ = (1, …, 1) and covariance Σ = I. In [1]:

import numpy

def getdata(seed):

n = 10 # data points

d = 50 # dimensionality of data m = numpy.ones([d]) # true mean

s = 1.0 # true standard deviation

<pre>    rstate = numpy.random.mtrand.RandomState(seed)
    X = rstate.normal(0,1,[n,d])*s+m
</pre>
return X,m,s

The following function computes the maximum likelihood estimator from a sample of the data assumed to be generated by a Gaussian distribution:

In [2]: def ML(X):

return X.mean(axis=0)

Implementing the James-Stein Estimator (10 P)

Based on the ML estimator function, write a function that receives as input the data (Xi)ni = 1 and the (known) variance σ2 of the generating distribution, and computes the James-Stein estimator

In [3]:

def JS(X,s):

# REPLACE BY YOUR CODE import solutions

m_JS = solutions.JS(X,s) ###

return m_JS

Comparing the ML and James-Stein Estimators (10 P)

We would like to compute the error of the maximum likelihood estimator and the James-Stein estimator for 100 different samples (where each sample consistsof10drawsgeneratedbythefunction getdata withadifferentrandomseed).Here,forreproducibility,weuseseedsfrom0to99.Theerror should be measured as the Euclidean distance between the true mean vector and the estimated mean vector.

Compute the maximum-likelihood and James-Stein estimations. Measure the error of these estimations.

Build a scatter plot comparing these errors for different samples.

</div>
</div>
<div class="layoutArea">
<div class="column">
‖ μˆ M L ‖ 2

</div>
</div>
</div>
</div>
<div class="page" title="Page 3">
<div class="section">
<div class="layoutArea">
<div class="column">
In [4]:

%matplotlib inline

### REPLACE BY YOUR CODE import solutions solutions.compare_ML_JS() ###

Part 2: Bias/Variance Decomposition (30 P)

In this part, we would like to implement a procedure to find the bias and variance of different predictors. We consider one for regression and one for classification. These predictors are available in the module utils.

utils.ParzenRegressor : A regression method based on Parzen window. The hyperparameter corresponds to the scale of the Parzen window. A large scale creates a more rigid model. A small scale creates a more flexible one.

utils.ParzenClassifier : A classification method based on Parzen window. The hyperparameter corresponds to the scale of the Parzen window. A large scale creates a more rigid model. A small scale creates a more flexible one. Note that instead of returning a single class for a given data point, it outputs a probability distribution over the set of possible classes.

Each class of predictor implements the following three methods:

__init__(self,parameter): Createaninstanceofthepredictorwithacertainscaleparameter. fit(self,X,T): Fitthepredictortothedata(asetofdatapoints X andtargets T). predict(self,X): Compute the output values arbitrary inputs X .

To compute the bias and variance estimates, we require multiple samples from the training set for a single set of observation data. To acomplish this, we utilizethe Sampler classprovided.Thesamplerisinitializedwiththetrainingdataandpassedtothemethodforestimatingbiasandvariance,whereits function sampler.sample() iscalledrepeatedlyinordertofitmultiplemodelsandcreateanensembleofpredictionforeachtestdatapoint.

Regression Case (15 P)

For the regression case, Bias, Variance and Error are given by:

Bias(Y)2 = (EY[Y − T])2 Var(Y) = EY[(Y − EY[Y])2] Error(Y) = EY[(Y − T)2]

Task: Implement the KL-based Bias-Variance Decomposition defined above. The function should repeatedly sample training sets from the sampler (as many times as specified by the argument nbsamples), learn the predictor on them, and evaluate the variance on the out-of-sample distribution given by X and T.

In [5]:

def biasVarianceRegression(sampler, predictor, X, T, nbsamples):

# ——————————– # TODO: REPLACE BY YOUR CODE

# ——————————– import solutions

<pre>    bias,variance = solutions.biasVarianceRegression(sampler, predictor, X, T, nbsamples=nbsamples)
</pre>
<pre>    # --------------------------------
</pre>
return bias,variance

Your implementation can be tested with the following code:

</div>
</div>
</div>
</div>
<div class="page" title="Page 4">
<div class="section">
<div class="layoutArea">
<div class="column">
In [6]:

import utils,numpy

%matplotlib inline utils.plotBVE(utils.Housing,numpy.logspace(-6,3,num=30),utils.ParzenRegressor,biasVarianceRegression,’Housing Reg ression’)

Classification Case (15 P)

We consider here the Kullback-Leibler divergence as a measure of classification error, as derived in the exercise, the Bias, Variance decomposition for such error is:

Bias(Y) = DKL(T | | R) Var(Y) = EY[DKL(R | | Y)] Error(Y) = EY[DKL(T | | Y)]

where R is the distribution that minimizes its expected KL divergence from the estimator of probability distribution Y (see the theoretical exercise for how it is computed exactly), and where T is the target class distribution.

Task: Implement the KL-based Bias-Variance Decomposition defined above. The function should repeatedly sample training sets from the sampler (as many times as specified by the argument nbsamples), learn the predictor on them, and evaluate the variance on the out-of-sample distribution given by X and T.

In [7]:

def biasVarianceClassification(sampler, predictor, X, T, nbsamples=25):

# ——————————– # TODO: REPLACE BY YOUR CODE

# ——————————– import solutions

<pre>    bias,variance = solutions.biasVarianceClassification(sampler, predictor, X, T, nbsamples=nbsamples)
</pre>
<pre>    # --------------------------------
</pre>
return bias,variance

Your implementation can be tested with the following code:

</div>
</div>
</div>
</div>
<div class="page" title="Page 5">
<div class="section">
<div class="layoutArea">
<div class="column">
In [8]:

import utils,numpy

%matplotlib inline utils.plotBVE(utils.Yeast,numpy.logspace(-6,3,num=30),utils.ParzenClassifier,biasVarianceClassification,’Yeast Cl assification’)

</div>
</div>
</div>
</div>
<div class="page" title="Page 6"></div>
<div class="page" title="Page 7"></div>
<div class="page" title="Page 8"></div>
