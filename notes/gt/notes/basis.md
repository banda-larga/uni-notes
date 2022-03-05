---
layout: post
title: Prima parte
---
Un **grafo** è una struttura algebrica. Un grafo è una coppia $$G=(V, E)$$ di insiemi (_nodi/vertici_ e _lati/archi_) tali che $$E \subset |V|^2$$ (tutte le coppie di elementi distinti di $$V$$).
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
\epsilon (G) = |E| \setminus |V|
$$

## Seconda parte

Andiamo a vedere altre strutture: 

Un **cammino** di lunghezza $$k$$ in un grafo $$G$$ è un sottografo $$P_k$$ che contiene $$k$$ archi e $$k+1$$ vertici, in cui gli archi $$e_1, \dots, e_k$$ e vertici $$k+1$$ tali che $$e_i = (v_{i-1}, v_i)$$. Questi vertici notiamo che possono avere neighbors (vicinati) ampi quanto vogliamo. 

Quindi è un grafo non vuoto $$P$$ della forma:

$$
V = \{x_0, x_1, \dots, x_k \} \quad E = \{x_{0}x_{1}, \dots, x_{k-1}x_k \}
$$

La sua _lunghezza_ è il numero dei lati che lo compongono. Se ha lunghezza $$k$$ lo denotiamo appunto con $$P_k$$. Può anche essere $$0$$, in quel caso $$P^0=K^1$$.

Due cammini sono _indipendenti_ se non contengono un vertice interno in comune.  

Un **ciclo** $$C_k$$ (in questo caso $$k$$-ciclo per la sua lunghezza) di lunghezza $$k \ge 3$$ è formato da un cammino $$P_{k-1}$$ (con $$k$$ vertici e $$k-1$$ archi) che può essere esteso in $$G$$ (in G trovo un arco per chiudere) includendo l'arco $$(v_{k-1}, v_0)$$. 

In un grafo $$G$$, il **calibro** (girth) $$g(G)$$ è la lunghezza del ciclo più breve (maggiore di 3 ovviamente). 

La lunghezza del ciclo più lungo è chiamata **circonferenza** del grafo $$G$$. 

Un lato che unisce due vertici di un ciclo ma che non è un lato del ciclo è detto **corda**.

Ci sono modi per capire le relazioni tra le quantità che caratterizzano il grafo? Un ciclo in un grafo biologico può avere una certa importanza. 

<span class="newthought">Prop 1</span>: Ogni $$G$$ con grado minimo 2 $$\delta (G) \ge 2$$, contiene un cammino di lunghezza pari al grado minimo e un ciclo di lunghezza almeno grado minimo più uno $$\delta (G) + 1$$. 

**dim**: Prendiamo un grafo $$G$$, prendo il più lungo cammino del grafo. Guardo i neighbors (vicinato) di $$v_k$$. Tutti i vicini del vertice $$v_k$$ devono stare sul cammino. Se ci fosse un vicino non sul cammino potremmo allungare il cammino con un nodo in $$V$$ e non sarebbe più quello di lunghezza massima. Quindi abbiamo detto che non possiamo allungare il cammino: $$N(v_k) \in P_k$$. Il grado di $$v_k$$, $$k \ge d(v_k) \ge \delta v_k$$. 

Andiamo a prendere il primo vertice che è un vicino di $$v_k$$ ($$(v_i, v_k) \in E$$). Abbiamo cardinalità di $$v_k$$ + 1 per tornare indietro, aggiungendo questo lato. $$C$$ è quindi lungo almeno $$\delta (G) + 1$$ altrimenti non posso chiudere il ciclo. 

E' semplice trovare questa relazione. 

Introduciamo ora il concetto di **distanza**. Su un oggetto come un grafo, essendo in una metrica non euclidea, non è così banale definire una distanza. Induciamo una nozione di distanza fra due vertici di un grafo. 

Preso $$G =(V, E)$$, $$\forall i, j \in V$$, allora $$d(i, j)$$ è la distanza. Se $$i, j$$ sono connessi in $$G$$ da almeno un cammino allora $$d(i, j)$$ è la lunghezza del cammino più breve, altrimenti (se il grafo non è connesso) la distanza è infinita $$d(i, j) = \inf$$. 

A questo punto possiamo definire il concetto di **diametro** sul grafo. Il diametro non è altro che la distanza massima fra tutte le coppie di vertici:

$$
diam(G) = \max_{i, j \in V} d(i, j)
$$

Il **raggio** del grafo ($$rad(G)$$) viene definito come per una circonferenza: 

$$
rad(G) = \min_{i \in V} \max_{j \in V} d(i, j)
$$

Se è un raggio ci aspettiamo una relazione simile (la metà del diametro). 

Il raggio è sicuramente più piccolo del diametro: $$rad(G) \le diam(G) \le ?$$. Chiamiamo $$x \in V$$ un vertice tc $$d(x, v) \le rad(G)$$ (il massimo è minimizzato su questo vertice). (x è tipo il centro del grafo, o meglio uno dei centri)

Prendiamo qualsiasi vertice $$u, v \in V$$:

$$
d(u, v) \le d(u, x) + d(x, v)
$$

Ciascun di questi elementi è più grande $$\gt rand(G)$$. Per qualsiasi coppia di vertici la loro distanza è più piccola di $$2 rad(G)$$. 

Cosa hanno a che fare raggio e diametro con i cicli? 

<span class="newthought">Fatto 1</span>: Ogni grafo $$G$$ con almeno un ciclo soddisfa (girth calibro) $$g(G) \lt 2 diam(G) + 1$$. C'è quindi un limite alla lunghezza del ciclo più breve di un grafo. 

**dim**: Prendiamo un grafo che contiene almeno un ciclo. Sia $$C$$ il ciclo di lunghezza minima $$g(G)$$. Prendiamo due vertici agli estremi opposti (tagliano il ciclo in due cammini il più possibile uguali). Assumo per assurdo che $$g(G) \ge 2 diam(G) + 2$$ sia falsa. 

A questo punto abbiamo due cammini $$P_1, P_2$$. Sono lunghi almeno $$diam(G) + 1$$. Però, guardando la distanza in $$G$$ dei due punti, abbiamo $$d(x, y) < diam(G)$$ per definizione di diametro. 

Non tutti gli archi di un cammino $$P$$ (di lunghezza minima) stanno su $$C$$ (il mio ciclo), se tutti gli archi stessero su $$C$$ avrei un modo per andare da $$x$$ a $$y$$ minore (non sarebbe un ciclo di lunghezza minima). Allora $$P$$ insieme al più breve tra $$P_1, P_2$$ forma un ciclo minore, posso quindi costruire un ciclo più breve di quello che ho assunto essere quello più breve, il che è contraddittorio. 

Un'altra cosa interessante è la **connettività** di un grafo. Un grafo è **connesso** se è non-vuoto e ogni coppia di vertici sono uniti da un cammino in $$G$$. 

Preso un grafo $$G$$ una _componente_ è un qualunque insieme massimale di vertici connessi (sottografo connesso). 

Un grafo $$G$$ è $$k$$-connesso se $$|V| > K$$ e $$\forall X,V$$ con $$|X| < K$$, il sottografo indotto da $$V \setminus X$$ è connesso. Qualsiasi grafo è $$0$$-connesso e sono $$1$$-connessi quelli semplicemente connessi (tranne $$K_1$$).

Il massimo intero $$k$$ t.c. $$G$$ è $$k$$-connesso è la connettività di $$G$$.

$$
\kappa(G)
$$

<span class="newthought">Teo 1</span>: Se $$G$$ non appartiene a $$K_0, K_1$$, cioè non è banale, allora la cardinalità di $$G$$, $$\kappa (G) \le \lambda (G) \le \delta (G)$$.
{% include marginnote.html id="mn-lambdag" note="(dove nella lezione si è usato $$|F|$$ al posto di $$\lambda (G)$$, qualsiasi insieme minimo di archi la cui rimozione sconnette il grafo $$\lambda (G)$$ è la connettività degli archi)" %}. 

Se prendiamo due cliques e taglio l'unico arco che le connette: $$|F| = 1$$ e $$\delta (G) = n+1$$. 

**dim**: $$\kappa (G) \le |F|$$ (e fissiamo F), lo mostriamo nei prossimi casi. Prendiamo $$G'=(V, E \setminus F)$$:

- $$G$$ ha un vertice $$v$$ che non è incidente con un lato in $$F$$ (non sta tra i sottografi connessi da $$F$$). Chiamiamo $$C$$ la componente di $$G'$$ che contiene $$v$$ e consideriamo l'insieme di vertici di $$C$$ che sono incidenti con un lato di $$F$$: $$V_C$$. Se rimuoviamo questi vertici, allora $$v$$ è disconnesso da ogni componente di $$G$$ (starà in un sottografo, componente, sconnesso dall'altra componente). 
Allora $$\kappa (G) \lt |V_C|$$. Dall'altro lato, nessun lato in $$F$$ Può avere entrambi i vertici in $$C$$ (altrimenti $$F$$ non è minimo): $$\kappa (G) \lt |V_C| \lt |F|$$. 

- $$G$$ è tale che tutti i vertici sono incidenti con qualche arco di $$F$$. Prendiamo un vertice random e sia $$C$$ la componente di $$G'$$ che lo contiene. Alcuni $$u \in N(v)$$ sono tali che $$(v, u) \in F$$. Gli altri nodi nel vicinato di $$v$$ devono per forza appartenere a $$C$$ ed essere inidenti con lati distinti di $$F$$ (altrimenti non è minimo).
Possiamo dire che $$d(v) \lt |F|$$, che implica $$d(v) = |F| = \delta (G)$$, perché sappiamo che $$d(v) \le |F|$$. Siccome rimuovere il vicinato di $$v$$ disconnette $$v$$, concludiamo che $$\kappa(G) \le \delta (G) = |F|$$.

## Alberi e Foreste 

Un grafo _aciclico_, cioè un grafo che non contiene cicli è detto **foresta**. 

Una foresta connessa è detta **albero**. {% include marginnote.html id="mn-albero" note="Ciò implica che una foresta è un grafo le cui componenti sono alberi." %}

I vertici con grado $$1$$ in un albero sono le sue **foglie**, gli altri sono i vertici interni. Ogni albero non banale ha una foglia, ad esempio le estremità di un cammino di lunghezza massima. 

Questo teorema caratterizza quali grafi sono alberi. 

<span class="newthought">Teo 2</span>: _i seguenti enunciati sono equivalenti per un grafo_ $$T$$:

- $$T$$ è un albero.
- ogni coppia di vertici in $$T$$ sono uniti da un unico cammino in $$T$$.
- $$T$$ è minimamente connesso, per esempio $$T$$ è connesso ma $$T-e$$ non lo è, per ogni lato $$e \in T$$
- $$T$$ è _aciclico massimale_, $$T$$ non contiene cicli ma $$T + xy$$ sì, $$\forall$$ due vertici in $$T$$ non adiacenti. 

Un **albero di copertura** (spanning tree) di un grafo connesso $$G =(V, E)$$ è un sottografo $$T=(V, E')$$ che è un albero.{% include marginnote.html id="mn-acopertura" note="Un'applicazione del teorema molto frequente è che ogni grafo connesso contiene un albero di copertura: o prendiamo il minimo sottografo connesso di copertura e usiamo la terza, o prendiamo un sottografo aciclico massimale e usiamo la quarta." %}

Quando $$T$$ è un albero di copertura di $$G$$, i lati in $$E(G) \setminus E(T)$$ sono le **corde** di $$T$$ in $$G$$. 

<span class="newthought">Cor 1</span>: Un grafo connesso con $$n$$ vertici è un albero sse ha $$n-1$$ vertici. 

**dim**: per induzione su $$i$$ si mostra che il sottografo coperto dai primi $$i$$ vertici ha $$i-1$$ vertici. Per $$i=n$$ questo dimostra l'implicazione. Al contrario, preso $$G$$ un qualsiasi grafo connesso con $$n$$ vertici e $$n+1$$ lati. Diciamo $$G'$$ un albero di copertura in $$G$$. Siccome $$G$$ ha $$n-1$$ lati dalla prima implicazione, concludiamo dicendo che $$G=G'$$. 

<span class="newthought">Cor 2</span>: Se $$T$$ è un albero e $$G$$ è un qualunque grafo con $$\delta (G) \ge |T| + 1$$ allora $$T \subseteq G$$, ad esempio $$G$$ ha un sottografo isomorfo a $$T$$. 