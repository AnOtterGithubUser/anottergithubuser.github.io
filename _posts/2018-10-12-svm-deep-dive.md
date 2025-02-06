---
title:  "SVM - deep dive"
date:   2018-10-12 16:12:23 +200
excerpt: "Support Vectors Machine has long been the go-to algorithm for machine learning. Deep dive into the remarkable power of this classifier."
toc: true
toc_label: SVM
toc_icon: ""
header:
  teaser: /assets/images/svm_icon.png
---

### Introduction

The Support Vector Machine is a large margin classifier i.e we are looking for the hyperplane that separates the data with the largest margin. Which means that for a binary classification problem where $y \in \{-1,1\}$ we want to find $f$ that maximizes $y_i f(x_i)$.     
      
Historically, it has been the first "kernel method" for pattern recognition. It has a number of advantages (fast, sparse solutions, ...) and is still one of the most popular algorithms in machine learning, often reaching state of the art performance.     
To understand this demonstration, you need a certain degree of familiarity with linear algebra, reproducing kernel hilbert space, and the representer theorem. I recommend the [course](http://members.cbio.mines-paristech.fr/~jvert/svn/kernelcourse/slides/master2017/master2017.pdf) of Jean Philippe VERT which I used extensively for this post.
       
### Primal formulation
        
We have already seen the $L2$ and logistic loss for the [linear](./2018-09-05-linear-regression-deep-dive.md) and [logistic regression](./2018-09-28-logistic-regression-deep-dive.md). For SVM we introduce the hinge loss:     
      
$$
\phi : x \rightarrow max(1-x,0)
$$
       
Since we are trying to maximize our margin $y_i f(x_i)$ the optimization problem that SVM solves is:
           
$$
\begin{align}
min_{f \in H} \frac{1}{N}\sum_i^{N} \phi (y_i f(x_i)) + \lambda ||f||_H^{2}
\end{align}
$$
     
The representer theorem helps us to get a form of $f$:     
      
$$
f(x) = \sum_i^{N} \alpha_i K(x_i, x)
$$
     
Now we need to find $(\alpha_i)_{i \in \{1,..,N\}}$ to solve the equation (1) that we can rewrite as:     
       
$$
\begin{equation}
min_{\alpha \in \mathbb{R}^N} \frac{1}{N} \sum_i^N \phi(y_i [K \alpha]_i) + \lambda \alpha^T K \alpha
\end{equation}
$$
          
This is a convex optimization problem in $\alpha$. However, the hinge loss is not smooth...
We can notice at this point that there is not constraint on the data being perfectly separable here, unlike many formulations we can find. If $x_i$ is misclassified, then $y_i f(x_i) < 0$ and $\phi(y_i f(x_i)) > 1$. Nothing keeps that from happening in our formulation, thus we will find a soft-margin SVM. This is actually a good thing since real data are rarely perfectly separable and a hard margin classifier would not find a solution in this case.       
          
Let's introduce slack variables to make our function smooth:     
      
$$
\begin{gather}
min_{\alpha \in \mathbb{R}^N, \xi \in \mathbb{R}^N} \frac{1}{N} \sum_i^N \xi_i + \lambda \alpha^T K \alpha   \\
s.t \quad \xi_i \geq \phi(y_i [K \alpha]_i)
\end{gather}
$$

Now the objective function is smooth but the constraints are not... Let's decompose them:     
     
$$
\begin{gather}
min_{\alpha \in \mathbb{R}^N, \xi \in \mathbb{R}^N} \frac{1}{N} \sum_i^N \xi_i + \lambda \alpha^T K  \alpha     \\
s.t \quad \xi_i \geq 0   \\
\xi_i \geq 1 - y_i [K \alpha]_i
\end{gather}
$$

Finally, our problem is well posed. The objective function is convex and smooth and constraints are smooth too. We can solve this equation with any optimization package !
However the dimension is quite large, $2N$. For a large dataset the optimization could be very slow...

### Dual formulation

To solve a constrained convex problem, the general approach is to use Lagrange multipliers and write the Lagrangian. Let's introduce $\mu, \eta \in \mathbb{R}^N$:     
      
$$
\begin{gather}
L(\alpha, \xi, \mu, \eta) = \frac{1}{N} \sum_i^N \xi_i + \lambda \alpha^T K \alpha - \sum_i^N \mu_i [y_i [K \alpha]_i + \xi_i-1] - \sum_i^N \eta_i \xi_i     \\
= \xi^T \frac{1}{N} + \lambda \alpha^T K \alpha - (diag(y)\mu)^T K \alpha - (\mu+\eta)^T \xi + \mu^T 1
\end{gather}
$$      
        
We want to minimize the Lagrangian. Let's write its gradient:     
      
$$
\begin{gather}\nabla_{\alpha} L = 2 \lambda K \alpha - K (diag(y)\mu) = K (2 \lambda \alpha - (diag(y)\mu))   \\
\nabla_{\xi} L = \frac{1}{N} - \mu - \eta
\end{gather}
$$    
        
Then the optimal $\alpha$ is:     
      
$$
\begin{equation}
\hat{\alpha} = \frac{1}{2 \lambda} (diag(y)\mu)
\end{equation}
$$     
      
and:     
       
$$
\begin{equation}
\mu + \eta = \frac{1}{N}
\end{equation}
$$     
           
We want to minimize the Lagrangian with respect to $\alpha$, $\xi$ and then maximize it with respect to the Lagrange multipliers to get the dual problem. We have the optimal value for $\alpha$. Now, since $L: \xi \rightarrow L(\xi)$ is a linear function, its minimum is $-\infty$ if the coefficient is non zero, meaning if $\mu + \eta \neq \frac{1}{N}$. Then if $\mu + \eta = \frac{1}{N}$:         
          
$$
\begin{equation}
q(\mu, \eta) = L(\hat{\alpha}, \xi, \mu, \eta) = \mu^T 1 - \frac{1}{4 \lambda} (diag(y) \mu)^T K (diag(y) \mu)
\end{equation}
$$       
         
For $q$ to take finite values, $0 \leq \mu \leq \frac{1}{N}$, therefore the dual formulation is:     
      
$$
\begin{gather}
max_{\mu} \mu^T 1 - \frac{1}{4 \lambda} (diag(y) \mu)^T K (diag(y) \mu)     \\
s.t \quad 0 \leq \mu \leq \frac{1}{N}
\end{gather}
$$    
              
Now what we really want are the coefficients $\alpha$ that fully parametrize the solution function according to the representer theorem. The link between $\alpha$ and $\mu$ is pretty straightforward, let's plug $\alpha$ in our formulation of the dual:       
       
$$
\begin{equation}
max_{\alpha} 2 \alpha^T \frac{1}{y} - \alpha^T K \alpha
\end{equation}
$$        
         
We ommited $\lambda$ since it is present in both terms. Also $\frac{1}{y} = y$ since $y_i \in \{-1, 1\}$. So our problem is:          
         
$$
\begin{gather}
max_{\alpha} 2 \alpha^T y - \alpha^T K \alpha     \\
s.t \quad 0 \leq \alpha^T (diag(y)) \leq \frac{1}{2 \lambda N}
\end{gather}
$$       
          
Finally ! We have a quadratic problem of one variable with a smooth objective function and a smooth constraint. Any quadratic solver package (I will use cvxopt) will do !

### Sparsity of the solution

Before implementing, I would like to highlight the complementary slackness conditions. Since we have strong duality we know that the constraint terms in the Lagrangian are zero at the feasible point (aka solution) $\hat{\alpha}$, therefore:     

$$
\begin{gather}
\mu_i [y_i [K \alpha_i - 1 + \xi_i] = 0    \\
\eta_i \xi_i = 0
\end{gather}
$$     
         
And that is extremely interesting ! You might have heard that SVM returns sparse solutions and only the support vectors are actually important to define the function. These conditions explain it. To see that, let's plug $\hat{\alpha}$ in them.       
         
$$
\begin{gather}
\hat{\alpha_i} (y_i [K \alpha]_i - 1 + \xi_i) = 0      \\
(\frac{y_i}{2 \lambda N} - \hat{\alpha_i}) \xi_i = 0
\end{gather}
$$        
          
If $ 0 \leq \hat{\alpha}_i \leq \frac{y_i}{2 \lambda N} $, then $\xi_i = 1 - y_i f(x_i)$ and $\xi_i = 0$, so:      
       
$$
\begin{gather}
y_i f(x_i) = 1
\end{gather}
$$      
          
This means that $x_i$ is a support vector!       
          
Now if $\hat{\alpha}_i = \frac{y_i}{2 \lambda N}$, this means that $\xi_i = 1 - y_i f(x_i)$ and thus:     
       
$$
\begin{equation}
y_i f(x_i) \leq 1
\end{equation}
$$    
              
So $x_i$ is inside the margin!       
But, you say, the points cannot be inside the margin, that ruins the principle of large-margin classifier, how can it be ? Well, remember when we introduced the slack variables $\xi_i$. I said that nothing prevent the points from being misclassified here. In fact, our formulation of the SVM is a soft-margin classifier.<br>
If $\hat{\alpha}_i = 0$, then $\xi_i = 0$, so $y_i [K \alpha]_i - 1 \geq 0$, which means:      
        
$$
\begin{equation}
y_i f(x_i) \geq 1
\end{equation}
$$         
            
These points are outside the margin. The form of $\hat{f}$ tells us that they do not contribute. Therefore only the support vectors and the points inside the margin characterize the solution function. Since most of the points will be outside of the margin, this leads to a sparse solution!

### An efficient implementation

Using `cvxopt` to compute the solution of the dual formulation would work but it would actually be very slow. We have to compute the kernel matrix which is rarely sparse, and the optimisation packages will usually have high memory requirements. The SMO (Sequential Minimal Optimisation) algorithm is a popular decomposition method in this case. It is described in this [article](https://emilemathieu.fr/posts/2018/08/svm/). I would recommand reading it and [Bottou et al](https://leon.bottou.org/publications/pdf/lin-2006.pdf) section 6, 7.1, and 7.2 to get the idea.
