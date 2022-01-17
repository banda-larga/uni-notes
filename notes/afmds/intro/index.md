---
layout: post
title: Introduzione
---

In questa introduzione iniziamo con un po' di fatti generali sul *data mining*.

## Cosa si intende per Data Mining

Nel 1990 il data mining era un concetto nuovo ed eccitante. Nel 2010 le persone hanno iniziato a parlare di *big data*. Oggi, il termine più popolare è *data science*. Comunque, per tutto questo tempo, il concetto è rimasto lo stesso: l'uso dell'hardware più potente e dei più efficienti algoritmi per risolvere i problemi nelle scienze, nel commercio, nella sanità, a livello governativo e in molti altri campi di interesse.

### Modellazione

Per i più, con il termine data mining si indica il processo che porta a creare un modello a partire dai dati, solitamente mediante *machine learning*. Più generalmente, l'obiettivo del data mining è un algoritmo. Ad esempio, nel trattare il locality-sensitive hashing - e algoritmi di stream-mining - non si fa riferimento ad un modello. Comunque, nella maggior parte delle applicazioni, la parte difficile è la creazione di quest'ultimo, e una volta che il modello è disponibile, l'algoritmo è immediato.{% include sidenote.html id="note_es1.1" note="Esempio 1.1: Consideriamo il seguente problema: dobbiamo determinare se certe email fanno parte di un attacco di phishing. L'approccio più comune è quello di costruire un modello sulle email di phishing, analizzando magari quelle che le persone hanno segnalato come tali, e cercando parole o frasi che appaiono frequentemente, come "Principe Nigeriano" o "Verifica il tuo Account". Il modello potrebbe consistere nell'assegnare dei pesi alle varie parole, con pesi positivi per quelle che appaiono frequentemente nelle email di phishing e negativi per quelle che non appaiono. A questo punto l'algoritmo è semplice. Applichiamo il modello ad ogni email, sommiamo i pesi delle parole nell'email e diciamo che l'email fa parte di una campagna di phishing se la somma è positiva (Detto ciò, trovare i pesi migliori non è un problema semplice, che verrà affrontato poi)." %}

### Machine Learning 

Alcuni vedono il data mining come sinonimo di machine learning. Non c'è dubbio che in certi casi il data mining usi algoritmi del machine learning. Nel machine learning si usano i dati come training set, per addestrare uno dei tanti algoritmi (support-vector machines, decision trees, hidden Markov models etc.). 

Ci sono situazioni dove usare i dati in questo modo ha senso. Il tipico caso in cui possiamo considerare il machine learning un buon approccio è quando non abbiamo un'idea ben definita di cosa i dati ci dicano nell'affrontare il problema che dobbiamo risolvere. Per esempio, non è molto chiaro cosa, dei film, li porti ad essere apprezzati dal pubblico e dai critici. Però, in questo caso, nella sfida di Netflix di realizzare un algoritmo per predire il punteggio di un film da parte degli utenti, il machine learning si è dimostrato molto efficace.

A parte questo, il machine learning si è provato non ottimale in situazioni in cui possiamo descrivere con più certezza gli *obiettivi* del mining. Un caso interessante è quello della startup *WhizBang! Labs*, che ha cercato di usare il machine learning per trovare i curriculum per le persone nel Web. Non riusciva a fare meglio di algoritmi progettati direttamente per cercare frasi ovvie e parole che appaiono in un tipico curriculum. In questo caso non c'era vantaggio nell'uso di tecniche di machine learning rispetto alla progettazione diretta di un algoritmo. 

Un altro problema con alcuni metodi di machine learning è che portano ad un modello che, per quanto accurato, non è *descrivibile*. In alcuni casi ciò non è importante, per esempio, se chiediamo a Google perché ha classificato certe email come spam, ci risponde con "sembra simile ad altri messaggi che le persone hanno identificato come spam", in altri, ad esempio se parliamo di una compagnia di assicurazioni, sì. 

### Approcci Computazionali alla Modellazione

In contrasto all'approccio statistico, gli informatici tendono a considerare il data mining come un problema algoritmico. In questo caso, il modello dei dati è semplicemente la risposta ad una domanda complessa sui dati.

Molti degli approcci alla modellazione possono essere descritti come:

- Riassumere i dati
- Estrarre le features più importanti ed ignorare il resto

### Riassumendo 

Si andranno a trattare problemi come *Web Mining*, {% include marginfigure.html id="bn" url="assets/img/johnsnow.png" description="Un famoso esempio di clustering a Londra interamente realizzato senza computer. Il medico John Snow, alle prese con
un’epidemia di colera, ha tracciato i casi su una mappa della città. I casi si sono raggruppati intorno ad alcune delle intersezioni delle strade. Queste intersezioni erano le posizioni dei pozzi contaminati; le persone che vivevano vicino a questi pozzi si ammalavano, mentre le persone che vivevano più vicino a pozzi non contaminati non si ammalavano. Senza la capacità di raggruppare i dati, la causa del colera non sarebbe stata scoperta." %}*Clustering*, tecniche per realizzare *Recommender Systems*, *Machine Learning* etc.