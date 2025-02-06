---
title:  "Logistic regression - deep dive"
date:   2018-09-28 10:21:54 +200
excerpt: ""
---
Logistic regression comes from the desire to model the log odds as linear functions. In the example of two classes classification, the model is presented as such:    
        
$$
\begin{equation}
log (\frac{p(y_i=0 | x_i, \theta)}{p(y_i=1 | x_i, \theta}) = \theta^T x_i \tag{1}
\end{equation}
$$     
     
Here I added the bias in $\theta$ and thus a constant term 1 in $x_i$.
From this equation we get:    
        
$$
\begin{gather}
log(\frac{p(y=0 | x, \theta)}{1 - p(y=0 |x, \theta)}) = \theta^T x  \tag{2} \\
p(y=0 | x, \theta) = \frac{e^{\theta^T x}}{1 + e^{\theta^T x}}  \tag{3} \\
p(y=0 | x, \theta) = \frac{1}{1 + e^{-\theta^T x}}  \tag{4}
\end{gather}
$$   
      
Now we have a set of samples ${x_1, x_2, ..., x_N}$ that we will denote $(x_i)$. Let's write the likelihood of these samples under the logistic assumption:     
     
$$
\begin{equation}
L(\theta) = \prod_{i=1}^N p(y_i | x_i, \theta)  \tag{5}
\end{equation}
$$    
       
We can write the conditional probability of Y on X as:    
        
$$
\begin{equation}
p(y_i | x_i, \theta) = p(y=0 | x_i, \theta)^{1-y_i} p(y=1 | x_i, \theta)^{y_i}  \tag{6}
\end{equation}
$$

The log-likelihood is usually easier to minimize. This case is no exception. Let's write the log-likelihood of our samples:      
       
$$
\begin{gather}
l(\theta) = \sum_i y_i log(1 - p(y_i=0 | x_i, \theta)) + (1-y_i)log(p(y_i=0|x_i, \theta))  \\
= \sum_i y_i (-\theta^T x_i - log(1 + e^{-\theta^T x_i})) - (1 - y_i) log(1 + e^{-\theta^T x_i})  \\
= \sum_i - y_i \theta^T x_i - log(1 + e^{-\theta^T x_i})
\end{gather}
$$     
        
This ends up to quite a simple form. However there is no closed form of the solution $\frac{\partial l}{\partial \theta} = 0$. We will have to use a gradient descent algorithm. We need the gradient then:     
          
$$
\begin{gather}
\frac{\partial l}{\partial \theta} = \sum_i -y_i x_i - (-x_i e^{-\theta^T x_i} \frac{1}{1 + e^{-\theta^T x_i}}) \\
= \sum_i -y_i x_i + x_i (1 - p(x_i, \theta)) \\
= \sum x_i (1 - p(x_i, \theta) - y_i)
\end{gather}
$$   
      
And thus our update schema is:     
      
$$
\begin{equation}
\theta_{new} = \theta_{old} - \alpha \frac{\partial l}{\partial \theta}(\theta_{old})  \tag{7}
\end{equation}
$$

Where $\alpha$ is the learning rate.     
Our gradient update follows from a very basic first order gradient descent algorithm. One could use a more advanced algorithm like [Newton-Raphson](http://en.wikipedia.org/wiki/Newton%27s_method).