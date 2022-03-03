---
layout: post
title: Prima parte
---
Un **grafo** è una coppia $$G=(V, E)$$ di insiemi (_nodi/vertici_ e _lati/archi_) tali che $$E \subset |V|^2$$.
(Un grafo con insieme di vertici $$V$$ è detto essere un grafo _su_ $$V$$).

Il numero di vertici (nodi) di un grafo è il suo **ordine**.
I grafi sono _finiti, infiniti, numerabili ..._ in base al loro ordine. 

Il grafo vuoto: $$(\emptyset, \emptyset) = \emptyset$$.
Un grafo di ordine $$0$$ o $$1$$ è detto _triviale_.

Un vertice $$v$$ è detto **incidente** ad un lato $$e$$ se $$v \in e$$. I due vertici incidenti con un lato sono le sue _estremità_. 

- $$E(X, Y)$$: insieme di tutti i lati $$X-Y$$ in un insieme $$E$$.
- $$E(v)$$ insieme di tutti i lati in $$E$$ incidenti a $$v$$.

$$x, y$$ di $$G$$ sono _adiacenti_ (vicini) se $$\{x, y\}$$ è un lato di $$G$$. Due lati sono adiacenti se hanno un estremo in comune.

Se tutti i vertici di $$G$$ sono adiacenti a due a due, $$G$$ è detto **completo** (cricca). Un grafo completo di $$n$$ vertici è un $$K^n$$, e $$K^3$$ è detto **triangolo**.

Coppie di vertici o lati non adiacenti sono detti **indipendenti**. Insiemi indipendenti di vertici sono chiamati **stabili**.

<span class="newthought">Omomorfismo</span>: siano $$G=(V, E)$$ e $$G'=(V', E')$$ due grafi. Una mappa $$\phi : V \to V'$$ è detto **omomorfismo** da $$G$$ a $$G'$$ se preserva l'adiacenza dei vertici, cioè se 

$$
\{\phi(x), \phi(y)\} \in E' \forall \{x, y\} \in E
$$

se per ogni vertice $$x'$$ nell'immagine di $$\phi$$, la sua inversa $$\phi^{-1}(x')$$ è un un insieme di vertici indipendenti in $$G$$.

<span class="newthought">Isomorfismo</span>: se $$\gamma$$ è **biettiva** e $$\gamma^{-1}$$ è un omomorfismo allora $$\gamma$$ è un **isomorfismo**. 

Un isomorfismo da $$G$$ in se stesso è detto _automorfismo_. {% include marginnote.html id="mn-invariants" note="Una mappa che prende grafi e assegna valori uguali a grafi isomorfi è detta **invariante**" %}

<span class="newthought">Sottografo</span>: se $$G'=(V', E')$$ è tale che $$V' \subseteq V$$ e $$E' \subseteq [V']\cap E$$ allora $$G'$$ è un **sottografo** di $$G$$ (G è **supergrafo**).{% include marginnote.html id="mn-subgraphs" note="Se $$G' \subseteq G$$ e $$G' \neq G$$, $$G$$ è un sottografo proprio. Se $$G' \subseteq G$$ e $$E' \subseteq [V']^2 \cap e$$, $$G$$ è un sottografo **indotto**." %}

### Grado

Prendiamo un grafo non vuoto $$G=(V, E)$$. Il vicinato di un vertice $$v$$ è indicato da $$N_{G}(v)$$.
Il **grado** di un nodo $$v$$, $$d_{G}(v)$$ è il numero di lati incidenti (la cardinalità di $$N(v)$$). Un vertice con grado zero è detto isolato. Il numero $$\delta (G) = \min \{d(v) | v \in V\}$$ è il **grado minimo**. Il **grado massimo** invece è indicato con $$\Delta (G) = \max \{d(v) | v \in V\}$$. 

Se tutti i vertici di $$G$$ hanno lo stesso grado $$k$$, $$G$$ è detto $$k$$-regolare. 

Grado medio:{% include marginnote.html id="mn-avgdegree" note="vediamo facilmente che $$\delta(G) \le d(G) \le \Delta(v)$$." %}

$$
d(G) = \frac{1}{|V|} \sum_{v \in V} d(v)
$$

Se vogliamo farci del male definiamo la **densità dei lati**. 

$$
\epsilon (G) = |E| \ |V|
$$

## Seconda parte

Andiamo a vedere altre strutture: 

Un **cammino** di lunghezza $$k$$ in un grafo $$G$$ è un sottografo $$P_k$$ che contiene $$k$$ archi e $$k+1$$ vertici, in cui gli archi $$e_1, \dots, e_k$$ e vertici $$k+1$$ tali che $$e_i = (v_{i-1}, v_i)$$. Questi vertici notiamo che possono avere neighbors (vicinati) ampi quanto vogliamo. 

Archi e vertici sono distinti. 

$$P_0 = ?$$, sappiamo che $$P_0 = K_1$$

Un **ciclo** $$C_k$$ di lunghezza $$k \ge 3$$ è formato da un cammino $$P_{k-1}$$ (con $$k$$ vertici e $$k-1$$ archi) che può essere esteso in $$G$$ (in G trovo un arco per chiudere) includendo l'arco $$(v_{k-1}, v_0)$$. 

In un grafo $$G$$, il **calibro** (girth) $$g(G)$$ è la lunghezza del ciclo più breve (maggiore di 3 ovviamente). 

La lunghezza del ciclo più lungo è chiamata **circonferenza** del grafo $$G$$. 

Possiamo dire: se io forzo una quantità creerò dei cicli/cammini lunghi? Ci sono modi per capire le relazioni tra le quantità che caratterizzano il grafo? Un ciclo in un grafo biologico può avere una certa importante. 

<span class="newthought">Fatto</span>: $$\forall G$$ con grado minimo 2 $$\delta (G) \ge 2$$, contiene un cammino di lunghezza pari al grado minimo e un ciclo di lunghezza almeno grado minimo più uno $$\delta (G) + 1$$. 

**dim**: Prendiamo un grafo $$G$$, prendo il più lungo cammino del grafo. Guardo i neighbors (vicinato) di $$v_k$$. Tutti i vicini del cammino $$P_k$$ devono stare sul cammino. Se ci fosse potremmo allungare il cammino. Siccome è di lunghezza massima, non possiamo allungare il cammino. $$N(v_k) \in P_k$$. Il grado di $$v_k$$, $$k \ge d(v_k) \ge \delta v_k$$. 

Dobbiamo dimostrare che esista un ciclo di lunghezza maggiore di 1. Andiamo a prendere il primo vertice che è un vicino di $$v_k$$. Almeno la cardinalità di $$v_k$$ + 1 per tornare indietro. $$C$$ è lungo almeno $$\delta (G) + 1$$ altrimenti non posso chiudere il ciclo. 

E' semplice trovare questa relazione. 

Introduciamo ora il concetto di **distanza**. Un oggetto come un grafo, essendo in una metrica non euclidea, non è così banale definire una distanza. Induciamo una nozione di distanza fra due vertici di un grafo. 

Preso $$G =(V, E)$$, $$\forall i, j \in V$$, allora $$d(i, j)$$ distanza. Se $$i, j$$ sono connessi in $$G$$ da almeno un cammino allora $$d(i, j)$$ è la lunghezza del cammino più breve, altrimenti (se il grafo è non è connesso allora la distanza è infinita) $$d(i, j) = \inf$$. 

A questo punto possiamo definire il concetto di **diametro** sul grafo. Il diametro non è altro che la distanza massima fra ogni coppia di vertici (max max)

$$
diam(G) = \max_{i, j \in V} d(i, j)
$$

Il **raggio** del grafo ($$rad(G)$$) viene definito come per una circonferenza: 

$$
rad(G) = \min_{i \in V} \max_{j \in V} d(i, j)
$$

Se è un raggio ci aspettiamo una relazione simile (la metà del diametro). 

Il raggio è sicuramente più piccolo del diametro: $$rad(G) \le diam(G) \le ?$$. Chiamiamo $$x \in V$$ un vertice tc $$d(x, v) \le rad(G)$$ (il massimo è minimizzato su questo vertice). (x è tipo il centro del grafo - uno dei centri)

Prendiamo qualsiasi vertice $$u, v \in V$$:

$$
d(u, v) \le d(u, x) + d(x, v)
$$

Ciascun di questi elementi è più grande $$\gt rand(G)$$. Per qualsiasi coppia di vertici la loro distanza è più piccola di $$2 rad(G)$$. 

Cosa hanno a che fare raggio e diametro con i cicli? 

<span class="newthought">Fatto:</span> $$\forall G$$ che ha almeno un ciclo, soddisfa (girth calibro) $$g(G) \lt 2 diam(G) + 1$$. C'è quindi un limite alla lunghezza del ciclo più breve di un grafo. 

**dim**: Prendiamo un grafo che contiene almeno un ciclo. Sia $$C$$ il ciclo di lunghezza minima $$g(G)$$. Prendiamo due vertici agli estremi opposti (tagliano il ciclo in due cammini il più possibile uguali). Assumo per assurdo che $$g(G) \ge 2 diam(G) + 2$$ sia falsa. 

A questo punto abbiamo due cammini $$P_1, P_2$$. Sono lunghi almeno $$diam(G) + 1$$. Però, guardando la distanza in $$G$$ $$d(x, y) \lt diam(G)$$ per definizione di diametro. 

Non tutti gli archi di un cammino $$P$$ (di lunghezza minima) stanno su $$C$$ (il mio ciclo), se tutti gli archi stessero su $$C$$ avrei un modo per andare da $$x$$ a $$y$$ minore. Allora posso comunque costruire un ciclo più breve di quello che ho assunto essere quello più breve. 

Un'altra cosa interessante è la **connettività** di un grafo. Un grafo è **connesso** se è non-vuoto se ogni coppia di vertici sono uniti da un cammino in $$G$$. 

Preso un grafo $$G$$ una _componente_ è un qualunque insieme massimale di vertici connessi (sottografo connesso). 

$$G$$ è $$k$$-connesso se $$|V| \gt K$$ e $$\forall X V$ con $$|X| \lt K$$, il sottografo indotto da $$V/X$$ è connesso. Qualsiasi grafo è $$0$$-connesso e sono $$1$$-connessi quelli semplicemente connessi (tranne $$K_1$$).

Il massimo intero $$k$$ t.c. $$G$$ è $$k$$-connesso è la connettività di $$G$$.

$$
\kappa(G)
$$

<span class="newthought">Teo:</span> Se $$G$$ non appartiene a $$K_0, K_1$$, cioè non è banale, allora la cardinalità di $$G$$, $$k(G) \le \lambda(G) \le \delta (G)$$ (dove nella lezione $$|F|$$ a posto di $$\lambda(G)$$ è qualisasi insieme minimo di archi la cui rimozione sconnette il grafo. $$\lambda(G) è la connettività degli archi$$). 

Se prendiamo due cliques e taglio l'unico arco che le connette, $$|F| = 1$$ e $$\delta(G) = n+1$$. 

**dim**: $$\kappa (G) \le |F|$$. Fissiamo F. Abbiamo due casi:

- $$G$$ ha un vertice $$v$$ che non è incidente a $$F$$. (non sta tra i sottografi connessi da $$F$$)

  Consideriamo l'insieme di vertici incidenti a $$F$$: $$V_c$$.

  Se rimuoviamo questi vertici, rimuoviamo anche gli archi di F, e v starà in un sottografo (componente) sconnesso dall'altra componente. La connettività, dato che rimuovendoli sconnetto il grafo, dev'essere non più della cardinalità di $$V_c$$: $$\kappa (G) \lt |V_c| \lt |F|$$. 

- $$G$$ è tale che tutti i vertici sono incidenti con qualche arco di $$F$$. 

  Possiamo dire che $$d(v) \lt |F|$$, sapendo che $$\kappa (G) \lt d(v)$$ allora $$\forall v$$, $$d(v) = \delta (G) = |F|$$.