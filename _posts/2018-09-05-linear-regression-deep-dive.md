---
title:  "Linear regression - deep dive"
date:   2018-09-05 14:43:52 +200
---
In this tutorial we will suppose that we have a matrix of samples $X$ and the corresponding labels $Y$
and we want to find the coefficients $w$ such that $Y = w^T X$.      
         
Basically, we want to characterize a function $f$ such that:    
     
$$
\begin{equation}
f: X \rightarrow w_0 + \sum_{j=1}^p X_j w_j  \tag{1} 
\end{equation}
$$      
     
If the samples $X$ were not drawn from a linear distribution, we would not be able to find such function. This is a simple model and the data
are rarely linearly separable.    
However it is still useful since it is a simple expressive model that requires few parameters. We need to find the hyperplan that fits the data as accurately as possible.       
       
Let's suppose we have a dataset of $((x_i, y_i))_{i \in \{1..N\}}$      
Then we would need a loss function to tell us how good our solution is. The most popular method is the least squares where the loss function is the
residual sum of squares:     
    
$$
\begin{equation}
L: X,Y,w \rightarrow \sum_{i=1}^N (y_i- f(x_i))^2    \tag{2} 
\end{equation}
$$    
     
The problem is then:     
$$
\begin{equation}
min_{w} L(X, Y, w)   \tag{3} 
\end{equation}
$$
           
Actually, we can write $L(X,Y,w)$ in matricial form:    
       
$$
\begin{equation}
L(X,Y,w) = (y - X w)^T (y - X w)   \tag{4} 
\end{equation}
$$     

We have a quadratic problem which we can differenciate
with respect to $w$:    
       
$$
\begin{gather}
\frac{\partial L(X,Y,w)}{\partial w} = -2 X^T (y - Xw)   \tag{5}    \\  \\ 
\frac{\partial^2 L(X,Y,w)}{\partial w \partial w^T} = 2 X^T X    \tag{6} 
\end{gather}
$$       
         
If $X$ has full rank, we can set the first derivative to $0$ and we obtain the unique solution:     
$$
\begin{equation}
\hat{w} = (X^T X)^{-1} X^T y   \tag{7} 
\end{equation}
$$     
Then our function $f$ is:    
        
$$
\begin{equation}
f: X \rightarrow X (X^T X)^{-1} X^T y   \tag{8} 
\end{equation}
$$      
               
Now if $X$ does not have full rank, $X^T X$ is singular and the solution is not unique.
On a geometric point of view, linear regression is a projection on the space spanned by the columns of $X$. In the case where $X$ does not have full rank, there are some redundancies between its columns. A usual way to solve this problem is to drop the redundant columns and get back to the full
rank case.    
We may also use the [Moore-Penrose pseudo inverse](https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse) in practice when we implement this solution.