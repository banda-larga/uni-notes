---
layout: post
title: JAX
---

# Cos'è JAX

Jax è una libreria Python (duh). E' una libreria simile a _NumPy_ ma con alcuni accorgimenti in più. E' sviluppata avendo come obiettivo l'efficienza, cosa che le permette di essere molto performante.

Jax usa XLA (Accelerated Linear Algebra - che permette di migliorare la velocità del codice ed avere anche altri miglioramenti anche a livello memoria, ad esempio con BERT aumentano le prestazioni di _7x_ e rende possibile aumentare la batch fino a _5x_) per andare a compilare il codice Numpy e permetterne l'esecuzione su GPUs e TPUs. 

Il codice viene compilato "under the hood", con le chiamate che vengono compilate just-in-time ed eseguite. 

Jax permette di compilare anche le proprie funzioni in kernel ottimizzati per XLA, utilizzando una sola funzione. 


```python
import jax.numpy as jnp
from jax import grad, jit, vmap
```


### Moltiplicazione fra Matrici

Una delle differenze principali fra NumPy e JAX è il modo in cui si generano numeri random (vediamo poi bro). 


```python
key = random.PRNGKey(0)
x = random.normal(key, (10,))
print(x)
```


Moltiplichiamo ora due matrici molto grandi:


```python
size = 3000
x = random.normal(key, (size, size), dtype=jnp.float32)
# jax usa funzioni asincrone di default (vedi as. dispatch)
%timeit jnp.dot(x, x.T).block_until_ready() 

# 489 ms ± 2.08 ms per loop
```


JAX NumPy ha funzioni che funzionano su Array regolari:


```python
import numpy as np
x = np.random.normal(size=(size, size)).astype(np.float32)
%timeit jnp.dot(x, x.T).block_until_ready()

# 435 ms ± 1.73 ms per loop
```


E' più lento perché deve trasferire i dati sulla GPU ogni volta. Possiamo assicurarci che un array sia on device con [`device_put()`](https://jax.readthedocs.io/en/latest/_autosummary/jax.device_put.html#jax.device_put).


```python
from jax import device_put

x = np.random.normal(size=(size, size)).astype(np.float32)
x = device_put(x)
%timeit jnp.dot(x, x.T).block_until_ready()

# 428 ms ± 899 µs per loop
```


Il comportamento di [`device_put()`](https://jax.readthedocs.io/en/latest/_autosummary/jax.device_put.html#jax.device_put) è lo stesso della funzione jit(lambda x: x) ma è più veloce. 

### Funzioni 

Avendo una GPU - o anche TPU queste funzioni saranno molto più performanti che in CPU. Ma JAX è molto più che NumPy che gira in GPU. Ha un po' di funzioni che sono molto utili quando si scrive codice per calcoli numerici:

- [`jit()`](https://jax.readthedocs.io/en/latest/_autosummary/jax.jit.html#jax.jit), rendere più veloce il codice
- [`grad()`](https://jax.readthedocs.io/en/latest/_autosummary/jax.grad.html#jax.grad), per le derivate
- [`vmap()`](https://jax.readthedocs.io/en/latest/_autosummary/jax.vmap.html#jax.vmap), vettorizzazione automatica o batching

## Jit()

JAX si può eseguire su quel che si vuole. Comunque, negli esempi finora visti JAX va ad eseguire un'operazione alla volta. Se si ha una sequenza di operazioni, possiamo utilizzare il decoratore `@jit` per compilare più operazioni insieme utilizzando XLA:


```python
def selu(x, alpha=1.67, lambda=1.05):
  return lambda*jnp.where(x>0, x, alpha*jnp.exp(x)-alpha)

x = random.normal(key, (100000,))
%timeit selu(x).block_until_ready()

# 2.94 ms ± 16.4 µs per loop
```


Possiamo rendere la funzione più veloce con `@jit`, che va a compilare la prima volta che la funzione viene chiamata e verrà quindi cachata per successive chiamate. 


```python
selu_jit = jit(selu)
%timeit selu_jit(x).block_until_ready()

# 746 µs ± 2.42 µs per loop
```


## Grad()

Oltre a valutare funzioni numeriche, vogliamo applicare trasformazioni. Una trasformazione è la [differenziazione automatica](https://en.wikipedia.org/wiki/Automatic_differentiation). 

```python
def sum_logistic(x):
  return jnp.sum(1.0 / 1.0 + jnp.exp(-x))

x_small = jnp.arange(3.)
derivative_fn = grad(sum_logistic)
print(derivative_fn(x_small))
# [0.25  0.19661197  0.10499357]
```

Possiamo comporre  [`grad()`](https://jax.readthedocs.io/en/latest/_autosummary/jax.grad.html#jax.grad) e [`jit()`](https://jax.readthedocs.io/en/latest/_autosummary/jax.jit.html#jax.jit) arbitrariamente.


```python
print(grad(jit(grad(jit(grad(sum_logistic)))))(1.0))
```


Per calcoli più complessi, ad esempio calcolare la Jacobiana:


```python
from jax import jacfwd, jacrev

def hessian(fun):
  return jit(jacfwd(jacrev(fun)))
```


## Auto-Vectorization con vmap()

JAX ha un'altra trasformazione che può risultare utile: [`vmap()`](https://jax.readthedocs.io/en/latest/_autosummary/jax.vmap.html#jax.vmap).

Ha una semantica simile al map di una funzione sugli assi di un array, ma è molto performante. Composta con [`jit()`](https://jax.readthedocs.io/en/latest/_autosummary/jax.jit.html#jax.jit), può essere veloce tanto quanto l'aggiungere le dimensioni a manina. 

Andiamo a fare un esempio semplice:


```python
mat = random.normal(key, (150, 100))
batched_x = random.normal(key, (10, 100))

def apply_matrix(v):
  return jnp.dot(mat, v)
```


Utilizzando ora la funzone possiamo andare a looppare sulle dimensioni


```python
# approccio naive
def nb_apply_matrix(v_batched):
  return jnp.stack([apply_matrix(v) for v in v_batched])

print('Naively batched')
%timeit nb_apply_matrix(batched_x).block_until_ready()
# 1.4 ms ± 766 ns per loop
```


```python
@jit
def b_apply_matrix(v_batched):
  return jnp.dot(v_batched, mat.T)

print('Manually batched')
%timeit b_apply_matrix(batched_x).block_until_ready()
# 10.9 µs ± 20.5 ns per loop
```


Supponiamo di avere una funzione più complessa, senza batching. Possiamo usare vmap per aggiungere batching automaticamente:


```python
@jit
def vmap_b_apply_matrix(v_batched):
  return vmap(apply_matrix)(v_batched)

print('Auto-vectorized with vmap')
%timeit vmap_b_apply_matrix(batched_x).block_until_ready()
# 32.5 µs ± 44.2 ns per loop
```


# JAX vs NumPy

JAX è un noto cantante italiano. Usare JAX con efficacia richiede sforzi mentali, bisogna spremere le meningi. 

- JAX fornisce un'interfaccia simile a NumPy.
- Attraverso la duck-typing, gli array JAX possono essere usati come quelli NumPy.
- Rispetto agli array NumPy, gli array JAX sono sempre _immutable_.


```python
import matplotlib.pyplot as plt 
import numpy as np

x_np = np.linspace(0, 10, 1000)
y_np = 2 * np.sin(x_np) * np.cos(x_np)
plt.plot(x_np, y_np)
```


```python
import jax.numpy as jnp

x_jnp = jnp.linspace(0, 10, 1000)
y_jnp = 2 * jnp.sin(x_jnp) * jnp.cos(x_jnp)
plt.plot(x_jnp, y_jnp)
```


Gli array sono differenti proprio a livello di tipiii:


```python
type(x_np)
# numpy.ndarray
```


```python
type(x_jnp)
# jax.interpreters.xla._DeviceArray
```


Gli array in JAX abbiamo detto che sono immutabili:


```python
# NumPy: mutable arrays
x = np.arange(10)
x[0] = 10
print(x)
# [10  1  2  3  4  5  6  7  8  9]

# JAX: immutable arrays
x = jnp.arange(10)
x[0] = 10
# TypeError: '<class 'jax.interpreters.xla._DeviceArray'>'
```


Se vogliamo cambiare gli elementi, in JAX dobbiamo creare una copia:


```python
y = x.at[0].set(10)
print(x)
print(y)
# [0 1 2 3 4 5 6 7 8 9]
# [10  1  2  3  4  5  6  7  8  9]
```


## To Jit or not to Jit 

- Di default JAX esegue le operazioni una alla volta, in sequenza
- Usando il decoratore just-in-time, le sequenze di operazioni possono essere ottimizzate ed eseguite insieme
- Non tutto il codice JAX può essere compilato, gli array devono avere dimensioni statiche e conosciute al tempo della compilazione

Prendiamo una funzione che normalizza le righe di una matrice 2D:


```python
import jax.numpy as jnp

def norm(X):
  X = X - X.mean(0)
  return X / X.std(0)
```


```python
from jax import jit
norm_compiled = jit(norm)
```


```python
np.random.seed(1701)
X = jnp.array(np.random.rand(10000, 10))
```


```python
%timeit norm(X).block_until_ready()
%timeit norm_compiled(X).block_until_ready()

# 100 loops, best of 3: 4.3 ms per loop
# 1000 loops, best of 3: 452 µs per loop
```


Se invece andiamo a scrivere una funzione le cui dimensioni dell'input array non sono conosciute a compile time 


```python
def get_negatives(x):
  return x[x < 0]

x = jnp.array(np.random.randn(10))
get_negatives(x)

jit(get_negatives)(x)
# IndexError
```


Se cerchiamo di decorare la funzione manco fosse la cappella sistina con jit ci vien fuori un bell'errore: siamo definitivamente scemi. 

### Jit... ancora... e basta 

Se vogliamo vedere come la funzione è codificata in jax, la jaxpr per essere brevi e concisi, basta che usiamo `jax.make_jaxpr`:


```python
from jax import make_jaxpr

def f(x, y):
  return jnp.dot(x + 1, y + 1)

make_jaxpr(f)(x, y)
#{ lambda  ; a b.
#  let c = add a 1.0
#      d = add b 1.0
#      e = culo etc...
```


### Static vs Traced

Così come i valori possono essere static o traced, anche le espressioni idem. 

Operazioni statiche: compile-time, operazioni traced: run-time (in XLA).


```python
import jax.numpy as jnp
from jax import jit

@jit
def f(x):
  return x.reshape(jnp.array(x.shape).prod())

x = jnp.ones((2, 3))
f(x)
# ConcretizationTypeError: tracer value 
```


Anche se x è traced, x.shape è static. Quando usiamo l'array di jnp su un valore statico, diventa traced, e non possiamo usarlo in una funzione tipo `reshape()` che richiede un input statico. 

Un buon pattern è usare numpy per operazioni che devono essere statiche:


```python
from jax import jit
import jax.numpy as jnp
import numpy as np

@jit
def f(x):
  return x.reshape((np.prod(x.shape),))

f(x)
# DeviceArray([1., 1., 1., 1., 1., 1.], dtype=float32)
```

Per questo quando si usa JAX ci serve anche NumPy. Va usato un po' di tutto di questi tempi. 

# Machine Learning con JAX

JAX è essenzialmente legato al paradigma funzionale. Ama la purezza infinita nelle funzioni.

```python
state = 0

def impura(x):
  return x + g 
```

Se andiamo ad eseguire questa due volte e la variabile globale cambia nel mentre, la funzione utilizzando jit viene cachata e restituisce la somma con lo stato cachato. 

```
print(jit(impura)(3.))
# 3.
state = 3 # andiamo a cambiarla
print(jit(impura)(3.))
# 3.
```

Anche se si vogliono generare numeri random, mentre il PRNG in NumPy è stateful, in JAX no.. te pareva

```python
seed = 0
state = jax.random.PRNGKey(seed)
state1, state2 = jax.random.split(state)
```

## Stateful to Stateless

Usiamo questo pattern per rendere una classe da Stateful a Stateless:

```python
class Stateful:
  state = state
  def stateful_method(
    *args, 
    **kwargs
  ) -> Output:
```

```python
class Stateless:
  def stateful_method(
    state, 
    *args, 
    **kwargs
  ) -> (Output, State):
```

## PyTree

Come JAX gestisce i gradienti. 

```python
f = lambda x, y, z, w = x**3 + y**2 + sqrt(z) + w
x, y, z, w = [1.]*4
dfdx, dfdy, dfdz, dfdw = grad(
  f, 
  argnums=(0, 1, 2, 3) # dobbiamo esplicitarli
)(x, y, z, w)

w -= w*dfdw
```

Questo non scala a 175 Miliardi di parametri ahah.

Siccome andiamo a wrappare i nostri parametri in strutture più complesse, pythree permette di trovare i gradienti in maniera più semplice. 

Avendo una struttura un po' più nested, possiamo andare a chiamare una funzione sul PyTree che ci dice quali sono le foglie:

```python
leaves = jax.tree_leaves(pytree)
```

Se vogliamo manipolare le foglie del nostro albero:

```python
jax.tree_map(lambda x: x**3, pytree)

# se abbiamo più pytrees
pytree2 = pytree
jax.tree_multimap(lambda x, y: x+y, pytree, pytree2)
# somma e ci restituisce un pytree
```

Devono avere la stessa struttura per il multimap. 

## Semplice NN

```python
# inizializziamo
def init_mlp_params(layer_w):
  for n_in, n_out in zip(layer_w[:-1], layer_w[1:]):
    params.append(
      dict(
        weights = np.random.normal(size = (n_in, n_out)) ** np.sqrt(2 / n_in), 
        biases = np.ones(shape=(n_out,))
      )
    )
    return params 

params = init_mlp_params([1, 128, 128, 1])
print(jax.tree_map(lambda x: x.shape, params))

# forward
def forward(params, x):
  *hidden, last = params
  for layer in hidden:
      x = jax.nn.relu(jnp.dot(x, layer['weights']) + layer['biases'])

# MSE loss
def loss(params, x, y):
  return jnp.mean((forward(params, x) - y) ** 2)

# update dei parametri
lr = 0.0001
@jit 
def update(params, x, y):
  grads = jax.grad(loss)(params, x, y)
  # SGD 
  return jax.tree_multimap(
      lambda p, g: p - lr*g, params, grads
  )

# addestriamo su una funzione seno
xs = np.random.normal(size=(128, 1))
ys = np.sin(xs)

EPOCHS = 2000
for _ in range(EPOCHS):
  params = update(params, xs, ys)
    
plt.scatter(xs, ys)
plt.scatter(xs, forward(params, xs), label = 'predictions')
plt.legend()
```
