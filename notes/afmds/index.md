---
layout: post
title: Algorithms for Massive Datasets
---

{% include maincolumn_img.html src='assets/imgs/op.png' description='' %}

## <center>Mathematical Preliminaries</center> 

## Linear Algebra

### Vector Spaces

**Vector spaces** are sets (the elements are called vectors) on which two operations are defined: vectors can be added (_sum_) and they can be multiplied by real numbers (_scalars_). More generally vector spaces can be defined over any Field. We take $$\mathbb{R}$$ to avoid any diversion into abstract algebra. {% sidenote "sidenote-uno" "This is a random sidenote" %}

A set of vectors $$v_1, \dots, v_n \in V$$ is said to be **linearly independent** if:

$$
\alpha_1v_1 + \dots + \alpha_nv_n = 0 \implies \alpha_1 + \dots + \alpha_n = 0
$$

The **span** of vectors in the set is the set of all vectors that can be expressed as a linear combination of them. If a set of vectors is linearly independent and its span is the whole of $$V$$, those vectors are said to be a **basis** of V. 

### Euclidean Space

With $$\mathbb{R}^n$$ we denote the Euclidean space. The vectors in this space consist of $$n$$-tuples of real numbers:

$$
\mathbf{x} = (x_1, x_2, \dots, x_n)
$$

It will be useful to think of them as $$n \times 1$$ matrices (or _column vectors_).

Addition and scalar multiplication are defined component-wise on vectors.

### Linear Maps

A **linear map** is a function $$T:V \to W$$, where $$V$$ and $$W$$ are vector spaces, that satisfies:

- $$T(\mathbf{x} + \mathbf{y}) = T\mathbf{x} + T\mathbf{y}$$ $$\forall \mathbf{x}, \mathbf{y} \in V$$
- $$T(\alpha\mathbf{x}) = \alpha T \mathbf{x}$$ $$\forall \mathbf{x} \in V$$, $$\alpha \in \mathbb{R}$$

A linear map from $$V$$ to itself is called a **linear operator**.

In algebraic terms, a linear map is also called a **homomorphism** of vector spaces, it preserves vector spaces' two main operations, addition and scalar multiplication. An invertible homomorphism (the inverse must be also a homomorphism) is called an **isomorphism**. If an isomorphism from $$V$$ to $$W$$ exists, $$V$$ and $$W$$ are said to be **isomorphic**. Isomorphic spaces are the same in terms of their algebraic structure. 

### The matrix associated to a Linear Map 

To represent vectors and to manipulate them, we use **matrices**. 

Suppose $$V$$ and $$W$$ to be finite-dimensional (Is finite-dimensional if it is spanned by a finite number of vectors), $$\mathbf{v}_{1}, \dots, \mathbf{v}_{n}$$ and $$\mathbf{w}_{1}, \dots, \mathbf{w}_{m}$$ are basis for the vector spaces, and $$T:V \to W$$ a linear map. Then, the matrix of $$T$$, with entries $$A_{ij}$$ where $$i=1, \dots, m$$, $$j=1, \dots, n$$ is defined by:

$$
T\mathbf{v}_j = A_{1j}\mathbf{w}_1 + \dots + A_{mj}\mathbf{w}_m
$$

That is, the _j_-th column of $$\mathbf{A}$$ consists of the coordinates of $$T\mathbf{v_j}$$ in the chosen basis for $$W$$. 

Every matrix $$A \in \mathbb{R}^{n \times m}$$ induces a linear map $$T: \mathbb{R}^n \to \mathbb{R}^m$$ given by:
$$
T\mathbf{x} = \mathbf{A}\mathbf{x}
$$
If $$\mathbf{A} \in \mathbb{R}^{n \times m}$$, its _transpose_ $$\mathbf{A}^T \in \mathbb{R}^{n \times m}$$ is given by $$(\mathbf{A}^T)_{ij} = \mathbf{A}_{ji}$$. 

### Metric Spaces

Metrics generalize the notion of distance. 

A metric on a set $$S$$ is a function $$d: S \times S \to \mathbb{R}$$ with its properties (positivity, symmetry and triangle inequality). 

A key motivation for metrics is that they allow limits to be defined for mathematical objects other than real numbers. A sequence converces to the limit _tot_ if for any $$\epsilon$$ ... $$d(x_n, x) \lt \epsilon $$ ...

### Normed Spaces 

Norms generalize the notion of length from Euclidean space. 

A **norm** on a real vector space is a function:

$$
\lVert \cdot \rVert : V \to \mathbb{R}
$$

A vector space endowed with a norm is called a **normed vector space** (or simly a **normed space**).

Any norm on $$V$$ induces a distance metric on $$V$$:

$$
d(\mathbf{x}, \mathbf{y}) = \lVert \mathbf{x} - \mathbf{y} \rVert
$$

So we can say that any normed space is also a metric space. If a normed space is _complete_ with respect to the distance metric induced by its norm it is a **Banach space**.

On $$\mathbb{R}^n$$ we have some specific norms:

- $$\lVert \mathbf{x} \rVert_1 = \sum_{i = 1}^{n} |x_i|$$
- $$\lVert \mathbf{x} \rVert_2 = \sqrt{\sum_{i = 1}^{n} x_i^2}$$
- $$\lVert \mathbf{x} \rVert_p = (\sum_{i = 1}^{n} |x_i|^p)^{1/p}$$
- $$\lVert \mathbf{x} \rVert_\infty = \max_{1 \le i \le n} |x_i|$$

### Inner Product Spaces

An **inner product** on a real vector space is a function $$\langle \cdot, \cdot \rangle : V \times V \to \mathbb{R}$$ satisfying:

- $$\langle \mathbf{x}, \mathbf{y} \rangle \ge 0 \Leftrightarrow \mathbf{x} = \mathbf{o}$$
- $$\langle \mathbf{x} + \mathbf{y}, \mathbf{z}\rangle = \langle \mathbf{x}, \mathbf{z}\rangle + \langle \mathbf{y}, \mathbf{z}\rangle$$ and $$\langle \alpha\mathbf{x}, \mathbf{y}\rangle = \alpha\langle \mathbf{x}, \mathbf{y} \rangle$$
- $$\langle\mathbf{x}, \mathbf{y}\rangle =\langle \mathbf{y}, \mathbf{x} \rangle$$

A vector space endowed with an inner product is called an **inner product space**. Any inner product on V induces a norm on V:

$$
\lVert \mathbf{x} \rVert = \sqrt{\langle \mathbf{x}, \mathbf{x} \rangle}
$$

Therefore, any inner product space is also a normed space (and also a metric space). If a normed space is _complete_ with respect to the distance metric induced by its inner product it is a **Hilbert space**. 

Two vector are **orthogonal** if $$\langle \mathbf{x}, \mathbf{y} \rangle = 0$$. Orhogonality generalize the notion of perpendicularity on Euclidean space. 

If two orthogonal vectors have also unit length, they are **orthonormal**.



## Calculus and optimization

Sometimes we have a function to minimize (**objective function**), which is a scalar function of several variables. 

### Extrema 

Optimization is mainly about finding **extrema**. When defining extrema, it is useful to consider the set of inputs over which we are optimizing (this set is called _feasible set_). If this set (we call it $$\chi \subseteq \mathbb{R}^d$$) is the entire domain of the function being optimized, the problem is **unconstrained**. Otherwise it is **constrained** ahah. 

### Gradients 

It is important. Yep bro. Gradients generalize derivatives to scalar functions of several variables. The gradient of $$f:\mathbb{R}^d \to \mathbb{R}$$, denoted with:

$$
\nabla f = \begin{bmatrix} \frac{\partial f}{\partial x_1}\\ \vdots \\ \frac{\partial f}{\partial x_n} \end{bmatrix}, \quad \nabla_i f =  \frac{\partial f}{\partial x_i}
$$

Gradients: $$\nabla f(\mathbf{x})$$ points in the direction of **steepest ascent**, $$- \nabla f(\mathbf{x})$$ points in the direction of **steepest descent**. We will use it frequently when minimizing a function via _gradient descent_.

### Jacobian

The **Jacobian** of $$f:\mathbb{R}^n \to \mathbb{R}^m$$ is a matrix of first-order partial derivatives:

$$
\mathbf{J}_f = \begin{bmatrix} \frac{\partial f_1}{\partial x_1} \dots \frac{\partial f_1}{\partial x_n}\\ \vdots \quad \ddots \quad \vdots \\ \frac{\partial f_m}{\partial x_1} \dots \frac{\partial f_m}{\partial x_n} \end{bmatrix}, \quad (\mathbf{J}_f)_{ij} =  \frac{\partial f}{\partial x_i}
$$

### Hessian 

The **Hessian** (The hessian is used in some optimization algos such as **Newton's method**) of $$f: \mathbb{R}^d \to \mathbb{R}$$ is a matrix of second-order partial derivatives:

$$
(\mathbf{H}_f)_{ij} \equiv \frac{\partial^{2} f}{\partial x_{i} \partial x_{j}}, \quad (\nabla^2)
$$

### Conditions for local minima

<span class="newthought">Prop:</span> If $$\mathbf{x}$$ is a local minimum of _f_ and _f_ is continuously differentiable in a neighborhood of $$\mathbf{x}$$, then $$\nabla f(\mathbf{x}) = \mathbf{0}$$. 

Vanishing gradient is necessary for an extremum: if $$\nabla f(\mathbf{x})$$ is non-zero, there exists a small step $$\alpha \gt 0$$ such that $$f(\mathbf{x} - \alpha \nabla f(\mathbf{x}))) \lt f(\mathbf{x})$$. Subtracting gradient is called a **descent direction**. Points where the gradient vanishes are called **stationary points**. There exists also **saddle points**, where the gradient vanishes but there is no local extremum. 

## 1st Notebook Analysis

### Visualize multi-variable functions

We will analyze real-valued functions having two or more real arguments. 

Starting with the simplest case: $$f: \mathbb{R}^2 \to \mathbb{R}$$. Such functions can be easily visualized in graphical form. 

Consider the function:
$$
g(x, y) = \frac{7}{5} e^{-\frac{(x-2)^2+(y-8)^2}{5}}+e^{-\frac{(x-7)^2+(y-5)^2}{10}}
$$
We can visualize it within $$[-2, 12]$$:{% include marginfigure.html id="bn" url="assets/imgs/plot3d.png" description="3D plot" %}

```python
# interactive, static with %matplotlib inline 
%matplotlib notebook

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
    
x_1d = np.linspace(-2.0, 12.0, 500)
y_1d = np.linspace(-2.0, 12.0, 500)
X, Y = np.meshgrid(x_1d, y_1d) # grid for x, y 

Z = 7/5 * np.exp(-.2 * ((X - 2)**2 + (Y - 8)**2)) + \
    np.exp(-.1*((X - 7)**2 + (Y - 5)**2))

ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', shade=False)
    
plt.show()
```

We can also visualize a contour plot via matplotlib's `contour`.

```python
%matplotlib inline

cp = plt.contour(X, Y, Z, 15)
plt.clabel(cp)
   
plt.show()
```

{% include marginfigure.html id="bn" url="assets/imgs/contourPlot.png" description="contour plot" %}

### Derivatives of multi-variable functions

```python
import sympy

x, y = sympy.symbols('x y')

z = 7/5 * sympy.exp( -.2 *((x - 2)**2 + (y - 8)**2)) + \
    sympy.exp(-.1*((x - 7)**2 + (y - 5)**2))

def gradient(expr, x_val, y_val):
    substitutions = [(x, x_val), (y, y_val)]
    return np.array([float(sympy.diff(expr, x).subs(substitutions)),
                     float(sympy.diff(expr, y).subs(substitutions))])
```

In a clever way:

```python
def gradient(expr, vars, vals):
    return np.array([float(sympy.diff(expr, var).subs(zip(vars, vals)))
                     for var in vars])
```

We can now watch the direction of the gradient:

```python
%matplotlib notebook

from IPython.display import clear_output

def gradient(expr, vars, vals):
    return np.array([float(sympy.diff(expr, var).subs(zip(vars, vals)))
                     for var in vars])

fig, ax = plt.subplots()

def onclick(event):

    ax.clear()
    
    cp = plt.contour(X, Y, Z, 15, zorder=0)
    plt.clabel(cp)
    
    if event is not None:
        grad = gradient(z, (x, y), (event.xdata, event.ydata))
        
        plt.arrow(event.xdata, event.ydata, grad[0], grad[1],
                  width=0.07, head_length=.2, zorder=1)
    

    clear_output(wait=True)
    display(ax.figure)

cid = fig.canvas.mpl_connect('button_press_event', onclick)

onclick(None)
plt.show()
```

Second-order derivatives can be used in order to search for local optima of a function. We have seen **Hessian** matrices, and now we define the **Hessian** of a fuction as the determinant of the Hessian matrix. 

We call **Laplacian** the sum (over $$n$$) of the second-order derivatives:

$$
\Delta f(x_1, \dots, x_n) = \sum_{i=1}^n \frac{\partial^2 f}{\partial x_i^2}
$$

It is possible to show that, given a point $$(x_1, \dots, x_n)$$ such that $$\nabla f(x_1, \dots, x_n) = 0$$:

- if $$\det \left( \mathrm H_f(x_1, \dots, x_n) \right) < 0$$, $$(x_1, \dots, x_n)$$ is a saddle point for $$f$$;
- if $$\det \left( \mathrm H_f(x_1, \dots, x_n) \right) > 0$$ and $$\Delta f(x_1, \dots, x_n) > 0$$, $$(x_1, \dots, x_n)$$ is a (either local or global) minimum of $$f$$;
- if $$\det \left( \mathrm H_f(x_1, \dots, x_n) \right) > 0$$ and $$\Delta f(x_1, \dots, x_n) < 0$$, $$(x_1, \dots, x_n)$$ is a (either local or global) maximum of $$f$$.


```python
def hessian(expr, x_val, y_val):
    hessian_matrix = sympy.Matrix([[sympy.diff(expr, v1, v2)
                                    for v1 in (x, y)]
                                    for v2 in (x, y)])
    return float(hessian_matrix.subs([(x, x_val), (y, y_val)]).det())

def laplacian(expr, x_val, y_val):
    l = [float(sympy.diff(expr, v, v).subs([(x, x_val), (y, y_val)]))
        for v in (x, y)]
                                 
    return sum(l)
```
<code>
def hessian(expr, x_val, y_val):
    hessian_matrix = sympy.Matrix([[sympy.diff(expr, v1, v2)
                                    for v1 in (x, y)]
                                    for v2 in (x, y)])
    return float(hessian_matrix.subs([(x, x_val), (y, y_val)]).det())

def laplacian(expr, x_val, y_val):
    l = [float(sympy.diff(expr, v, v).subs([(x, x_val), (y, y_val)]))
        for v in (x, y)]
                                 
    return sum(l)
</code>



# <center>Altro</center>

## Cosa si intende per Data Mining

Nel 1990 il data mining era un concetto nuovo ed eccitante. Nel 2010 le persone hanno iniziato a parlare di *big data*. Oggi, il termine più popolare è *data science*. Comunque, per tutto questo tempo, il concetto è rimasto lo stesso: l'uso dell'hardware più potente e dei più efficienti algoritmi per risolvere i problemi nelle scienze, nel commercio, nella sanità, a livello governativo e in molti altri campi di interesse.

### Modellazione

Per i più, con il termine data mining si indica il processo che porta a creare un modello a partire dai dati, solitamente mediante *machine learning*. Più generalmente, l'obiettivo del data mining è un algoritmo. Ad esempio, nel trattare il locality-sensitive hashing - e algoritmi di stream-mining - non si fa riferimento ad un modello. Comunque, nella maggior parte delle applicazioni, la parte difficile è la creazione di quest'ultimo, e una volta che il modello è disponibile, l'algoritmo è immediato.

### Machine Learning 

Alcuni vedono il data mining come sinonimo di machine learning. Non c'è dubbio che in certi casi il data mining usi algoritmi del machine learning. Nel machine learning si usano i dati come training set, per addestrare uno dei tanti algoritmi (support-vector machines, decision trees, hidden Markov models etc.). 

Ci sono situazioni dove usare i dati in questo modo ha senso. Il tipico caso in cui possiamo considerare il machine learning un buon approccio è quando non abbiamo un'idea ben definita di cosa i dati ci dicano nell'affrontare il problema che dobbiamo risolvere. Per esempio, non è molto chiaro cosa, dei film, li porti ad essere apprezzati dal pubblico e dai critici. Però, in questo caso, nella sfida di Netflix di realizzare un algoritmo per predire il punteggio di un film da parte degli utenti, il machine learning si è dimostrato molto efficace.

A parte questo, il machine learning si è provato non ottimale in situazioni in cui possiamo descrivere con più certezza gli *obiettivi* del mining. Un caso interessante è quello della startup *WhizBang! Labs*, che ha cercato di usare il machine learning per trovare i curriculum per le persone nel Web. Non riusciva a fare meglio di algoritmi progettati direttamente per cercare frasi ovvie e parole che appaiono in un tipico curriculum. In questo caso non c'era vantaggio nell'uso di tecniche di machine learning rispetto alla progettazione diretta di un algoritmo. 

{% include marginfigure.html id="bn" url="assets/imgs/johnsnow.png" description="Un famoso esempio di clustering a Londra interamente realizzato senza computer. Il medico John Snow, alle prese con
un’epidemia di colera, ha tracciato i casi su una mappa della città. I casi si sono raggruppati intorno ad alcune delle intersezioni delle strade. Queste intersezioni erano le posizioni dei pozzi contaminati; le persone che vivevano vicino a questi pozzi si ammalavano, mentre le persone che vivevano più vicino a pozzi non contaminati non si ammalavano. Senza la capacità di raggruppare i dati, la causa del colera non sarebbe stata scoperta." %}Un altro problema con alcuni metodi di machine learning è che portano ad un modello che, per quanto accurato, non è *descrivibile*. In alcuni casi ciò non è importante, per esempio, se chiediamo a Google perché ha classificato certe email come spam, ci risponde con "sembra simile ad altri messaggi che le persone hanno identificato come spam", in altri, ad esempio se parliamo di una compagnia di assicurazioni, sì. 

### Approcci Computazionali alla Modellazione

In contrasto all'approccio statistico, gli informatici tendono a considerare il data mining come un problema algoritmico. In questo caso, il modello dei dati è semplicemente la risposta ad una domanda complessa sui dati.

Molti degli approcci alla modellazione possono essere descritti come:

- Riassumere i dati
- Estrarre le features più importanti ed ignorare il resto

### Riassumendo 

Si andranno a trattare problemi come *Web Mining*, *Clustering*, tecniche per realizzare *Recommender Systems*, *Machine Learning* etc.
