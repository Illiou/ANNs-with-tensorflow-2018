# HW 1 Group 7 MLP & Backpropagation

## 1.1 and 1.2 Forward Pass
### First Layer:
Drive:

\begin{align*}
\vec{d}^{(1)} &= W^{(1)} \cdot \vec{x} + \vec{b} \\
&=  \begin{pmatrix}
-2 & 3 \\
2 & 1
\end{pmatrix} \cdot 
\begin{pmatrix}
2 \\
3 
\end{pmatrix} + \begin{pmatrix} 0 \\ 0 \end{pmatrix} \\
&= \begin{pmatrix}
(-2) \cdot 2 + 3 \cdot 3 + 0 \\
2 \cdot 2 + 1 \cdot 3 + 0 
\end{pmatrix} \\
&= \begin{pmatrix} 5 \\ 7 \end{pmatrix}
\end{align*}

Activation:
$$
\vec{a}^{(1)} = \sigma(\vec{d}^{(1)}) = {5 \choose 7}^2 = {25 \choose 49}
$$

### Final Layer:
Drive:

$$
d^{(2)} = \vec{w}^{(2)} \cdot \vec{a}^{(1)} = \begin{pmatrix} -2 & 1 \end{pmatrix} \cdot {25 \choose 49} = (-2) \cdot 25 + 1 \cdot 49 = -50 + 49 = -1
$$

Activation:

$$
y = \sigma(d^{(2)}) = (-1)^2 = 1
$$

### Illustration:

![](https://i.imgur.com/8yjKmCm.png)


## 1.3 Backpropagation 
#### a) Calculate the error:

$$L(t;y) = \frac{1}{2}(t-y)^2= \frac{1}{2}(3-1)^2 =\frac{1}{2}(2)^2=\frac{1}{2}\cdot4=2
$$

#### b) Derive the activation function:

\begin{align*}
\sigma(x)&=x^2 \\
\sigma'(x)&=2x
\end{align*}


#### c) Compute the partial derivatives of the loss function with regard to each weight, using the backpropagation alogrithm:

General rule *(see lecture slides)*:
\begin{gather*}
\frac{\partial L(\vec{t}; \vec{y})}{\partial w_{ji}^{(K)}} = \delta_j^{(K)} \cdot a_i^{(K-1)} \\
\text{where:} \\
\delta_j^{(K)} = \begin{cases} 
-(t_j - y_j) \cdot \sigma'(net_j^{(K)}) , & \text{if } K=L+1 \\
\sum\limits_k \delta_k^{(K+1)} \cdot w_{kj}^{(K+1)} \cdot \sigma'(net_j^{(K)}), &  \text{otherwise}
\end{cases}
\end{gather*}

Precalculations:
\begin{align*}
\delta_1^{(2)} &= -(t_1-y_1) \cdot \sigma'(net_1^{(2)}) = -(3 - 1) \cdot 2(-1) = -2 \cdot -2 = 4 \\
\delta_1^{(1)} &= \sum_{k=1}^1 \delta_k^{(2)} \cdot w_{k1}^{(2)} \cdot \sigma'(net_1^{(1)})= \delta_1^{(2)} \cdot w_{11}^{(2)} \cdot 2(5) = 4 \cdot (-2) \cdot 10 = -80 \\
\delta_2^{(1)} &= \sum_{k=1}^1 \delta_k^{(2)} \cdot w_{k2}^{(2)} \cdot \sigma'(net_2^{(1)})= \delta_1^{(2)} \cdot w_{12}^{(2)} \cdot 2(7) = 4 \cdot 1 \cdot 14  = 56
\end{align*}


Partial derivatives of the loss function with regard to each weight:
\begin{align*}
% w11 2
\frac{\partial L(\vec{t};\vec{y})}{\partial w_{11}^{(2)}} &= \delta_1^{(2)} \cdot a_1^{(1)} = 4 \cdot 25 = 100 \\
% w12 2
\frac{\partial L(\vec{t};\vec{y})}{\partial w_{12}^{(2)}} &= \delta_1^{(2)} \cdot a_2^{(1)} = 4 \cdot 49 = 196\\
% w11 1
\frac{\partial L(\vec{t};\vec{y})}{\partial w_{11}^{(1)}} &= \delta_1^{(1)} \cdot a_1^{(0)} = -80 \cdot 2 = -160\\
% w12 1
\frac{\partial L(\vec{t};\vec{y})}{\partial w_{12}^{(1)}} &= \delta_1^{(1)} \cdot a_2^{(0)} = -80 \cdot 3 = -240 \\
% w21 1
\frac{\partial L(\vec{t};\vec{y})}{\partial w_{21}^{(1)}} &= \delta_2^{(1)} \cdot a_1^{(0)} = 56 \cdot 2 = 112\\
% w22 1
\frac{\partial L(\vec{t};\vec{y})}{\partial w_{22}^{(1)}} &= \delta_2^{(1)} \cdot a_2^{(0)} = 56 \cdot 3 = 168 \\
\end{align*}


#### d) Compute the weight update for each weight
 
 General update rule:
 $$
 \theta_{new} = \theta_{old} - \gamma \nabla_{\theta}L(\vec{t};\vec{y})
 $$
 
 Individual weight updates:
 $$
 \begin{array}{ll}
 w_{11}^{(2)} = -2 - 0.01 \cdot 100 &= -3 \\
 w_{12}^{(2)} = 1 - 0.01 \cdot 196 &= -0.96 \\
 w_{11}^{(1)} = -2 - 0.01 \cdot (-160) &= -0.4 \\
 w_{12}^{(1)} = 3 - 0.01 \cdot (-240) &= 5.4 \\
 w_{21}^{(1)} = 2 - 0.01 \cdot 112& = 0.88 \\
 w_{22}^{(1)} = 1 - 0.01 \cdot 168 &= -0.68
 \end{array}
 $$
