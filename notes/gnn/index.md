---
layout: post
title: Machine Learning su Grafi
---

## Introduzione

In molti rami delle scienze, dalla fisica delle particelle alla biologia, i grafi vengono usati per modellare strutture complesse con oggetti ed interazioni.

<span class="newthought">Grafo</span>: un **grafo** è una coppia $$G = (V, E)$$ dove con $$V$$ si indicano i vertici o nodi e con $$E \subseteq V \times V$$ i lati (o archi) tra coppie di nodi.

Fatti, curiosità e altro sui grafetti si trovano sul [`Graph Theory`](https://diestel-graph-theory.com/) di Diestel o sul [`Graph Theory`]() di Bondy / Murty.

Metodi più semplici come _Node2Vec_ o _DeepWalk_ non catturano similarità strutturali e non vengono prese in considerazione le features dei nodi, lati o del grafo. 

<span class="newthought">Deep Representation Learning & Graph Neural Networks</span> 

es: in una rete sociale alcuni nodi identificano truffatori, invece altri identificano persone affidabili. Come troviamo gli _altri_ truffatori e gli altri _affidabili_?

Più nello specifico: preso uin grafo con labels su alcuni nodi, come assegnamo le labels agli altri nodi?

### correlazione

I nodi sono correlati: nodi vicini hanno lo stesso colore e appartengono alla stessa classe. 

Perché i nodi sono correlati? 

- Omofilia: la tendenza degli individui ad associarsi a quelli simili fra loro. (ricercatori dello stesso campo, uccelli)
- Influenza: le connessioni sociali possono influenzare le caratteristiche di una persona. (consigli musicali fra persone)

Come possiamo _strutturare_ questa _correlazione_ fra nodi per prevedere le labels dei nodi? 

<span class="newthought">Per associazione:</span> se sono connesso ad un nodo con label $$X$$, molto probabilmente ho anch'io la label $$X$$. La label può dipendere dalle _features_ di _v_, _labels_ nella neighborhood di _v_ e dalle _features_ nella neighborhood di _v_.

### Semi-Supervised Learning

Dato un grafo, qualche nodo labellato, dobiamo trovare la classe dei nodi rimanenti, prendendo come assunto che ci sia _omofilia_ nella rete. 

Rimando a Relational Classification, Iterative Classification, Correct & Smooth.

## Graph Neural Networks