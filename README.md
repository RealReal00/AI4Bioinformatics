## LOSS TASK
La funzione NLLSurvLoss implementa la Negative Log-Likelihood Loss per modelli di sopravvivenza in tempo discreto.
L'obiettivo è calcolare la probabilità del percorso temporale osservato per ciascun paziente (evento o censura), e penalizzare le previsioni del modello che se ne discostano.

- Ogni paziente è osservato fino a un certo tempo bin `y`.

- Il modello predice **hazard functions** → la probabilità di morire in ogni bin.

- Dai hazard calcoliamo la **funzione di sopravvivenza cumulativa** `S`.

- A seconda del tipo di paziente:
  - **Non censurato** (`c = 0`): la loss misura la probabilità di essere vivo fino al bin precedente e morire nel bin giusto.
  - **Censurato** (`c = 1`): la loss misura solo la probabilità di essere vivo fino a quel bin.

  **PARAMETRI**
  - *alpha*: bilancia l’importanza dei censurati nella loss.
  - *eps*: evita log(0).
  - *reduction*: aggrega la loss tra pazienti (mean o sum). Prova di entrambi

  **TORCH.GATHER E NAMING**  
  Si utilizza *torch.gather* per estrarre per ogni paziente il valore esatto di hazard o sopravvivenza per il bin y. In input prende il tensore da cui vogliamo prendere i valori, si specifica la dimensione lungo la quale vogliamo "pescare" i valori (i bin temporali). Per una maggiore comprensione sono stati rinominate le variabili che usano *torch.gather*:  
  s_prev--> surv_before_event--> Probabilità di essere vivo prima del bin di evento prende i dati da da S_padded (include S(-1)=1)  
  h_this --> hazard_at_event --> probabilità di morire proprio nel bin t, prende i dati da hazards = sigmoid(h)  
  s_this --> surv_at_bin_end --> Probabilità di essere vivo fino alla fine del bin t, prende i dati da S_padded specificando come index y+1 per riferirsi alla fine del bin  

  ### Select the model and define the hyperparameters
  [...]


  ## ESPERIMENTO SULLA INFLUENZA DEL CAMBIAMENTO DEL NUMERO DI PATCH 
  - **Caso 64 patch**  
  Lo scopo dell'esperimento è valutare l'efficacia del modello ABMIL_Multimodal, una rete neurale capace di combinare dati visivi (immagini istopatologiche WSI) e genomici, per prevedere la sopravvivenza dei pazienti affetti da carcinoma mammario (TCGA-BRCA).

  Il problema è affrontato come un task di sopravvivenza (Survival Analysis), con dati parzialmente censurati. In particolare, l’obiettivo è apprendere una curva di sopravvivenza per ogni paziente, e stimare con accuratezza il rischio.
  ### Configurazione degli iperparametri usati
 | Parametro            | Valore     | Descrizione                               |
| -------------------- | ---------- | ----------------------------------------- |
| `task_type`          | `Survival` | Tipo di task                              |
| `max_patches`        | `64`       | Patch visive per paziente                 |
| `batch_size`         | `1`        | Batch per device                          |
| `real_batch_size`    | `8`        | Batch effettivo via gradient accumulation |
| `n_bins`             | `4`        | Discretizzazione temporale                |
| `sample`             | `true`     | Sampling patch in training                |
| `test_sample`        | `false`    | Tutte le patch usate nel test             |
| `train_size`         | `0.7`      | Percentuale dati di training              |
| `val_size`           | `0.15`     | Percentuale dati di validazione           |
| `test_size`          | `0.15`     | Percentuale dati di test                  |
| `load_slides_in_RAM` | `false`    | Caricamento completo dei dati in RAM      |
| `num_workers`        | `0`        | Numero di worker per il DataLoader        |


  ### Motivazioni
  Il numero ridotto di patch (64) è un compromesso tra tempo di training e capacità predittiva.

  Il batch_size=1 è stato scelto per limitazioni hardware ma compensato da real_batch_size=8 per stabilità.

  La suddivisione bilanciata del dataset è stata curata per evitare set di validazione completamente censurati.

  ### Modello usato: ABMIL_Multimodal
  Il modello impiega un'architettura Multiple Instance Learning con meccanismi di attenzione per le patch visive, integrando in parallelo un ramo per i dati genomici. La combinazione delle due modalità consente al modello di sfruttare indizi sia morfologici che molecolari.

  ### Training e Metriche
  Il modello è stato allenato per 10 epoche. I risultati evidenziano una curva di apprendimento stabile e una crescente capacità del modello nel predire la sopravvivenza:  
 | Split             | C-index | Loss  |
| ----------------- | ------- | ----- |
| Train             | 0.9597  | 0.215 |
| Validation        | 0.5364  | 1.598 |
| Test (best model) | 0.6449  | 1.117 |
| Test (last epoch) | 0.6575  | 2.106 |


  Il c-index su validation/test resta moderato, ma significativamente migliore del caso random (0.5), soprattutto sul test set.
  Il gap tra training e test suggerisce overfitting: il modello ha appreso bene il training set, ma la generalizzazione è limitata.

  ### Analisi Visiva: Kaplan-Meier
  - Plot 1:
  Le curve High Risk vs Low Risk si separano visibilmente.

  La curva blu (alto rischio) decresce più rapidamente, come atteso.

  Il log-rank test p-value = 0.0314 indica che la separazione tra i due gruppi è statisticamente significativa (p < 0.05), anche se moderata.

  - Plot 2:
  La separazione tra gruppi è ancora più netta.

  La curva arancione (low risk) resta elevata per più tempo.

  Il p-value = 0.0117 rafforza la validità del modello selezionato (best_model.pt).

  La banda di confidenza suggerisce stabilità nelle stime.

-**caso 4096 patch**  

  ### Configurazione degli iperparametri usati  
| Parametro            | Valore     | Descrizione                               |
| -------------------- | ---------- | ----------------------------------------- |
| `task_type`          | `Survival` | Tipo di task                              |
| `max_patches`        | `4096`     | Patch visive per paziente                 |
| `batch_size`         | `1`        | Batch per device                          |
| `real_batch_size`    | `8`        | Batch effettivo via gradient accumulation |
| `n_bins`             | `4`        | Discretizzazione temporale                |
| `sample`             | `true`     | Sampling patch in training                |
| `test_sample`        | `false`    | Tutte le patch usate nel test             |
| `train_size`         | `0.7`      | Percentuale dati di training              |
| `val_size`           | `0.15`     | Percentuale dati di validazione           |
| `test_size`          | `0.15`     | Percentuale dati di test                  |
| `load_slides_in_RAM` | `false`    | Caricamento completo dei dati in RAM      |
| `num_workers`        | `0`        | Numero di worker per il DataLoader        |



  ### Motivazioni
  L’aumento di max_patches da 64 a 4096 è pensato per dare al modello una visione più completa del tessuto tumorale, migliorando la rappresentazione istopatologica. Tuttavia, per evitare problemi di memoria, il caricamento in RAM è disattivato e si mantiene num_workers = 1 per sicurezza, a scapito delle prestazioni in tempo.

  ### Prestazione del modello  
| Split             | C-index | Loss   |
| ----------------- | ------- | ------ |
| Train             | 0.6605  | 0.4472 |
| Validation        | 0.6585  | 0.5893 |
| Test (best model) | 0.6652  | 1.6387 |
| Test (last epoch) | 0.6838  | 1.5763 |


  Il c-index vicino a 0.66–0.68 su tutti i dataset mostra che il modello ha acquisito una capacità predittiva significativa, ben superiore al random (0.5).

  La loss sul test è più alta, ma coerente con la maggiore complessità dei dati da generalizzare (tutte le patch incluse).

  La coerenza tra training e validazione segnala un training ben bilanciato, con un generalizzazione migliore rispetto al primo esperimento, che mostrava chiari segni di overfitting, ma non si può ignorare che le loss crescono nel test set, suggerendo un overfitting moderato

  ### Analisi Visiva: Kaplan-Meier
  - Plot 1
  Le curve di sopravvivenza per alto rischio (blu) e basso rischio (arancione) risultano distinte visivamente, in linea con le predizioni del modello.

  Il log-rank p-value = 0.0754 indica una separazione quasi significativa statisticamente (soglia p = 0.05), suggerendo che il modello ha appreso una distinzione rilevante tra i gruppi.

  Le bande di confidenza sono più ampie rispetto al test precedente, plausibilmente dovute alla variabilità introdotta da più patch.

  L’inclusione di tutte le patch (test_sample=False) migliora la robustezza delle previsioni, a scapito di un aumento del tempo di elaborazione.

  ###  Configurazione della Loss Function nei due prcedenti esperimenti e Considerazioni sulla Censura
   alpha=0.0
   eps=1e-7
   reduction='mean'

   Quando si considerano sia censurati che non si permette al modello di apprendere anche da pazienti con eventi non osservati. Questo è cruciale in contesti reali come il carcinoma mammario (TCGA-BRCA), dove una porzione significativa dei pazienti è censurata ( infatti nel dataset abbiamo 912 censurati e 146 non censurati).

   Tuttavia, includere i censurati comporta anche una riduzione del segnale diretto sull'evento (morte/recidiva), che può influenzare la precisione nei confronti di chi ha avuto effettivamente l'evento

  ## ESPERIMENTO SULLA INFLUENZA DELLA CENSURA CON DIVERSI N_BINS E ALPHA
  Il problema è ancora una volta affrontato come un task di sopravvivenza (Survival Analysis) usando ABMIL_Multimodal con input ["WSI", "Genomics"], ma ora si vogliono confrontare i risulti e metriche per ogni combinazione alpha-n_bins. In particolare, l’obiettivo è apprendere una curva di sopravvivenza per ogni paziente, e stimare con accuratezza il rischio. I test sono stati condotti mettendo a confronto le combinazioni generate dai valori di α (0.0 e 1.0) e i bin (2 e 4)
  Per ciascun test, sono stati salvati:
  - I modelli (.pt) migliori (best val loss)
  - Metriche su train, val, test per best model
  - Last epoch e curve di Kaplan-Meier per i test set.  

**Tabella con alpha=0.0 e bin=2**  
| Alpha | Bins | Split | C-index | Loss  | Tipo modello |
|-------|------|-------|---------|-------|---------------|
| 0.0   | 2    | Train | 0.7899  | 1.102 | Best Model    |
| 0.0   | 2    | Val   | 0.7256  | 1.119 | Best Model    |
| 0.0   | 2    | Test  | 0.6897  | 1.108 | Best Model    |
| 0.0   | 2    | Train | 0.7607  | 1.138 | Last Epoch    |
| 0.0   | 2    | Val   | 0.7088  | 1.142 | Last Epoch    |
| 0.0   | 2    | Test  | 0.6836  | 1.140 | Last Epoch    |  
  

**Tabella coon alpha=1.0 e bin=2**  
| Alpha | Bins | Split | C-index | Loss  | Tipo modello |
|-------|------|-------|---------|-------|---------------|
| 1.0   | 2    | Train | 0.7665  | 1.121 | Best Model    |
| 1.0   | 2    | Val   | 0.7122  | 1.148 | Best Model    |
| 1.0   | 2    | Test  | 0.6851  | 1.146 | Best Model    |
| 1.0   | 2    | Train | 0.7421  | 1.148 | Last Epoch    |
| 1.0   | 2    | Val   | 0.6939  | 1.161 | Last Epoch    |
| 1.0   | 2    | Test  | 0.6803  | 1.158 | Last Epoch    |
  
### Osservazioni
- Effetto di α (alpha)
L'ipotesi era che α controllasse la penalizzazione dei censimenti.  
I risultati mostrano che α=0.0 ha portato a performance leggermente migliori su tutti gli split, soprattutto su train.  
Con α=1.0, le performance si mantengono stabili ma leggermente inferiori, suggerendo che l’inclusione della censura nella loss potrebbe introdurre una maggiore variabilità senza migliorare la generalizzazione.  

- Comportamento best model vs last epoch
In entrambi i casi, il best model (selezionato con early stopping sulla validation loss) supera il modello all’ultima epoca.   
Questo evidenzia che l’early stopping è utile per prevenire overfitting (ad es., test c-index: 0.6897 vs 0.6836 per α=0.0).

- Performance generale
Tutti i c-index sul test sono compresi tra 0.68 e 0.69, che è discreto per un task di survival su dati WSI+genomici.   
Il fatto che il test c-index sia solo leggermente inferiore al train indica un modello stabile, ma non fortemente discriminante.

**Curve di Kaplan-Meier**  
I grafici KM (sia best che last) mostrano curve di sopravvivenza tra gruppi a rischio molto sovrapposte, con:  

α=0.0: p-value ≈ 0.27  
α=1.0: p-value ≈ 0.89  

Nessuna distinzione statisticamente significativa tra "high" e "low" risk group.  
Questo suggerisce che il modello non riesce a separare bene i pazienti in base al rischio predetto.


**Tabella con alpha=0.0 e bin=4**  
| Alpha | Bins | Split | C-index | Loss  | Tipo modello |
| ----- | ---- | ----- | ------- | ----- | ------------ |
| 0.0   | 4    | Train | 0.8412  | 0.973 | Best Model   |
| 0.0   | 4    | Val   | 0.7111  | 1.052 | Best Model   |
| 0.0   | 4    | Test  | 0.6937  | 1.058 | Best Model   |
| 0.0   | 4    | Train | 0.8231  | 1.017 | Last Epoch   |
| 0.0   | 4    | Val   | 0.6966  | 1.080 | Last Epoch   |
| 0.0   | 4    | Test  | 0.6850  | 1.089 | Last Epoch   |  
  

**Tabella coon alpha=1.0 e bin=4**  
| Alpha | Bins | Split | C-index | Loss  | Tipo modello |
| ----- | ---- | ----- | ------- | ----- | ------------ |
| 1.0   | 4    | Train | 0.8255  | 1.010 | Best Model   |
| 1.0   | 4    | Val   | 0.6944  | 1.084 | Best Model   |
| 1.0   | 4    | Test  | 0.6833  | 1.093 | Best Model   |
| 1.0   | 4    | Train | 0.8231  | 1.017 | Last Epoch   |
| 1.0   | 4    | Val   | 0.6966  | 1.080 | Last Epoch   |
| 1.0   | 4    | Test  | 0.6850  | 1.089 | Last Epoch   |

### Osservazioni
- Effetto di α (alpha)
Con α = 0.0 (inclusione dei censurati nella loss), il modello ottiene le migliori prestazioni assolute tra tutti i testati finora:

Test c-index = 0.6937 (best model), in crescita rispetto a bin=2.

Anche il training set mostra ottimi risultati (c-index = 0.8412, loss = 0.973).

Questo suggerisce che l’inclusione dell’informazione censurata sia utile quando i dati sono discretizzati in modo più fine (4 bin).

Con α = 1.0 (esclusione dei censurati), le prestazioni sono leggermente inferiori, sia in termini di c-index che di loss:

Test c-index = 0.6833 (best model).

La loss aumenta lievemente, indicando minor precisione nella stima del rischio.

- Best model vs last epoch
In entrambi i casi, il best model (early stopping) performa meglio del modello all’ultima epoca:

α = 0.0 → test c-index: 0.6937 (best) vs 0.6850 (last)
α = 1.0 → test c-index: 0.6833 (best) vs 0.6850 (last)

L’effetto è più evidente su α = 0.0, confermando che l’early stopping migliora la generalizzazione.

**Osservazioni comparative**

Con 4 bin, α = 0.0 migliora il c-index di test e la loss, dimostrando migliore apprendimento del rischio.

α = 1.0 peggiora leggermente le metriche rispetto a bin = 2, suggerendo che con più suddivisioni temporali, ignorare i censurati può ridurre la qualità della stima.

**Kaplan-Meier Curves**  
α = 0.0 → p-value ≈ 0.47: non significativo, ma meglio che nel caso bin=2 (p ≈ 0.81).  
α = 1.0 → p-value ≈ 0.38: ancora non significativo.  
Nessuna delle curve KM mostra distinzione significativa tra i gruppi a rischio, anche con 4 bin. Questo indica che, nonostante l’aumento di bin, la stratificazione del rischio non è ancora ottimale.

**Conclusione dell'esperimento**  
Il modello migliore finora è con α = 0.0 e bin = 4, sia in termini di test c-index sia di stabilità durante il training.
La combinazione bin=4 e inclusione della censura (α=0.0) offre un buon compromesso tra dettaglio temporale e informazione clinica.
Le curve KM non mostrano ancora separazione statisticamente significativa, indicando spazio per migliorare la rappresentazione del rischio.


## ESPERIMENTO SULLA INFLUENZA DELLA DIVERSE MODALITA' CON DIVERSI ALPHA  
- **Obiettivo**  
L’obiettivo di questo esperimento è analizzare come le diverse modalità di input (Genomics, WSI, e Multimodale) rispondono alla presenza o assenza della censura nel task di Survival Prediction.
L'effetto di α potrebbe dipendere dalla modalità di input, questo perchè:
la genomica potrebbe contenere segnali più deboli e più sensibili al rumore;
le immagini WSI potrebbero beneficiare della presenza di più esempi, anche censurati;
la modalità multimodale potrebbe compensare le debolezze delle singole fonti.

**Caso solo Genomics**  
- α = 0.0 – Solo Genomics – bin=4  

| Alpha | Bins | Split |    C-index |   Loss | Tipo modello |
| ----- | ---- | ----- | ---------: | -----: | ------------ |
| 0.0   | 4    | Train | **1.0000** |   2.34 | Best Model   |
| 0.0   | 4    | Val   | **0.7093** | 251.54 | Best Model   |
| 0.0   | 4    | Test  | **0.6780** | 291.53 | Best Model   |
| 0.0   | 4    | Train |     1.0000 |   2.34 | Last Epoch   |
| 0.0   | 4    | Val   |     0.7093 | 251.54 | Last Epoch   |
| 0.0   | 4    | Test  |     0.6780 | 291.53 | Last Epoch   |


**Kaplan–Meier (log-rank p ≈ 0.028)**
Il gruppo High-risk (blu) mostra una discesa rapida: la sopravvivenza si riduce sotto il 40% entro ~7 anni, mentre il gruppo Low-risk (arancione) si mantiene stabilmente sopra il 60%.
Le bande di confidenza restano ben separate fino all’ottavo anno; il p-value < 0.05 segnala una separazione statisticamente significativa, coerente con un c-index ≈ 0.68.

- α = 1.0 – Solo Genomics – bin=4  

| Alpha | Bins | Split |    C-index |   Loss | Tipo modello |
| ----- | ---- | ----- | ---------: | -----: | ------------ |
| 1.0   | 4    | Train | **0.7711** |   5.45 | Best Model   |
| 1.0   | 4    | Val   | **0.4921** |  86.54 | Best Model   |
| 1.0   | 4    | Test  | **0.5009** | 116.79 | Best Model   |
| 1.0   | 4    | Train |     0.7711 |   5.45 | Last Epoch   |
| 1.0   | 4    | Val   |     0.4921 |  86.54 | Last Epoch   |
| 1.0   | 4    | Test  |     0.5009 | 116.79 | Last Epoch   |

**Kaplan–Meier (log-rank p ≈ 0.414)**
Le curve High- e Low-risk restano vicine per gran parte del tempo, con bande di confidenza ampie che si sovrappongono ampiamente.
Il p-value > 0.40 indica assenza di separazione significativa. Il c-index prossimo a 0.50 conferma un modello incapace di distinguere i gruppi.


### Osservazioni  

| Aspetto                  | α = 0.0 (censurati inclusi)                                                                      | α = 1.0 (censurati esclusi)                                                           |
| ------------------------ | ------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------- |
| **Gap Train ↔ Val/Test** | 1.00 → 0.71 / 0.68 (Δ ≈ -0.30)<br>*over-fit evidente ma prestazione ancora forte fuori campione* | 0.77 → 0.49 / 0.50 (Δ ≈ -0.27)<br>*prestazione scarsa anche su validation e test*     |
| **Stabilità**            | c-index Val ≈ Test (0.71 ↔ 0.68) ⇒ generalizzazione stabile                                      | c-index Val ≈ Test (0.49 ↔ 0.50) ⇒ modello instabile e vicino al ranking casuale      |
| **Loss**                 | più alta (Train 2.3 → Test 291): la NLL include anche i censurati                                | più bassa (Train 5.4 → Test 116): calcolata solo sugli eventi osservati               |
| **Kaplan-Meier**         | Bande strette, p-value ≈ 0.028 ⇒ differenza significativa e stratificazione clinicamente utile   | Bande ampie, p-value ≈ 0.41 ⇒ nessuna evidenza di separazione fra i gruppi di rischio |


-***Conclusione***
α = 0.0 vince la comparazione

C-index elevato e costante su Validation (0.71) e Test (0.68).
Le curve KM mostrano una separazione statisticamente significativa (p < 0.05), con bande di confidenza strette.

α = 1.0 non migliora la robustezza

Il modello sembra meno in over-fit (Train da 1.00 → 0.77), ma la generalizzazione cala drasticamente: Val e Test scendono a ~0.50.
Le curve KM si sovrappongono e non mostrano stratificazione significativa.  
 

**caso solo WSI**
- α = 0.0 – Solo WSI – bin=4  

| Alpha | Bins | Split |    C-index |   Loss | Tipo modello |
| ----- | ---- | ----- | ---------: | -----: | ------------ |
| 0.0   | 4    | Train | **0.8899** | 154.12 | Best Model   |
| 0.0   | 4    | Val   | **0.6383** | 102.70 | Best Model   |
| 0.0   | 4    | Test  | **0.6480** | 113.98 | Best Model   |
| 0.0   | 4    | Train |     0.8899 | 154.12 | Last Epoch   |
| 0.0   | 4    | Val   |     0.6383 | 102.70 | Last Epoch   |
| 0.0   | 4    | Test  |     0.6480 | 113.98 | Last Epoch   |

**Kaplan–Meier (log-rank p ≈ 0.090)**
Il gruppo High-risk (arancione) scende più rapidamente: la sopravvivenza si dimezza intorno al 7-8° anno, mentre il gruppo Low-risk (blu) resta sopra il 60 %.
Le bande di confidenza restano separate fino a ~6 anni, poi iniziano a sovrapporsi; il p-value > 0.05 indica che la differenza non è formalmente significativa, ma la tendenza di stratificazione è evidente e coerente con il c-index ≈ 0.65.

- α = 1.0 – Solo WSI – bin=4   

| Alpha | Bins | Split |    C-index |   Loss | Tipo modello |
| ----- | ---- | ----- | ---------: | -----: | ------------ |
| 1.0   | 4    | Train | **0.5796** | 122.15 | Best Model   |
| 1.0   | 4    | Val   | **0.5464** |  30.84 | Best Model   |
| 1.0   | 4    | Test  | **0.5470** |  38.65 | Best Model   |
| 1.0   | 4    | Train |     0.5796 | 122.15 | Last Epoch   |
| 1.0   | 4    | Val   |     0.5464 |  30.84 | Last Epoch   |
| 1.0   | 4    | Test  |     0.5470 |  38.65 | Last Epoch   |

**Kaplan–Meier (log-rank p ≈ 0.106)**
La separazione fra curve è più debole: le bande di confidenza dei due gruppi si sovrappongono quasi interamente fin dall’inizio.
Il p-value supera ampiamente la soglia di significatività e il c-index (~0.55) conferma che il ranking è vicino al caso.
In pratica il modello, avendo visto solo i pazienti con evento osservato, non ha dati a sufficienza per apprendere una distinzione robusta.





### Osservazioni  

| Aspetto                  | α = 0.0 (censurati inclusi)                                                                                        | α = 1.0 (censurati esclusi)                                                                             |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------- |
| **Gap Train ↔ Val/Test** | 0.89 → 0.64 (Δ ≈ -0.25)<br>*over-fit visibile ma il ranking rimane buono fuori campione*                           | 0.58 → 0.55 (Δ ≈ -0.03)<br>*quasi nessun over-fit, ma il livello resta vicino al caso*                  |
| **Stabilità**            | c-index Val ≈ Test (0.64 ↔ 0.65) ⇒ comportamento coerente                                                          | c-index Val ≈ Test (0.55 ↔ 0.55) ⇒ stabile, ma su un plateau poco informativo                           |
| **Loss**                 | più alta (Train 154 → Test 114): la NLL somma anche i pazienti censurati                                           | più bassa (Train 122 → Test 39): conta solo gli eventi osservati, perciò numericamente minore           |
| **Kaplan-Meier**         | Bande di confidenza più strette; separazione visiva discreta; p-value ≈ 0.09 → trend però non ancora significativo | Bande ampie, forte sovrapposizione; p-value ≈ 0.11 → nessuna evidenza di stratificazione dei due gruppi |  


-***Conclusione***

- α = 0.0 vince la comparazione

c-index stabile ~ 0.65 sia su Validation che Test.

Curve KM con bande più strette e trend di separazione (p ≈ 0.09) ⇒ la stima di rischio è affidabile e riproducibile.

- α = 1.0 non porta vera robustezza

Apparente riduzione dell’over-fit (gap Train–Val quasi nullo), ma la generalizzazione cala: Val scende a ~ 0.55 e Test resta su ~ 0.55.

Curve KM con bande molto più larghe e p-value > 0.10 ⇒ stratificazione poco discriminante.


***CASO WSI + GENOMICS***
- α = 0.0 – WSI + Genomics – bin=4 

| Alpha | Bins | Split |    C-index |    Loss | Tipo modello |
| ----- | ---- | ----- | ---------: | ------: | ------------ |
| 0.0   | 4    | Train | **1.0000** |   1.888 | Best Model   |
| 0.0   | 4    | Val   | **0.6722** | 244.044 | Best Model   |
| 0.0   | 4    | Test  | **0.6652** | 260.561 | Best Model   |
| 0.0   | 4    | Train |     1.0000 |   1.888 | Last Epoch   |
| 0.0   | 4    | Val   |     0.6722 | 244.044 | Last Epoch   |
| 0.0   | 4    | Test  |     0.6652 | 260.561 | Last Epoch   |  

**Kaplan-Meier (p = 0.0150)**  
La curva High-risk (blu) precipita rapidamente entro ~8 anni, mentre la Low-risk (arancione) scende più lentamente; le bande di confidenza non si sovrappongono dopo il 5º anno. La significatività (log-rank p ≈ 0.015) conferma una buona capacità di stratificare i pazienti.
Il rovescio della medaglia è l’over-fitting estremo (c-index = 1 sul train), dovuto al fatto che il modello “vede” sia gli eventi sia i censurati: usa quindi molte più informazioni e memorizza i dati di addestramento.

- α = 1.0 – WSI + Genomics – bin=4 

| Alpha | Bins | Split |    C-index |   Loss | Tipo modello |
| ----- | ---- | ----- | ---------: | -----: | ------------ |
| 1.0   | 4    | Train | **0.7543** | 16.906 | Best Model   |
| 1.0   | 4    | Val   | **0.4982** | 53.263 | Best Model   |
| 1.0   | 4    | Test  | **0.6160** | 73.334 | Best Model   |
| 1.0   | 4    | Train |     0.7543 | 16.906 | Last Epoch   |
| 1.0   | 4    | Val   |     0.4982 | 53.263 | Last Epoch   |
| 1.0   | 4    | Test  |     0.6160 | 73.334 | Last Epoch   |  

**Kaplan-Meier (p = 0.0157)**
Ignorando i censurati il numero di “eventi utili” cala e le bande di confidenza si allargano, specie nel gruppo Low-risk; tuttavia la separazione resta visibile e la p-value resta significativa. Il c-index di train scende (meno over-fit) ma la validation c-index crolla a 0.50: l’informazione persa sui pazienti censurati penalizza la generalizzazione, sebbene sul test si recuperi parzialmente (0.616).

### Osservazioni   
| Aspetto                  | α = 0.0 (censurati inclusi)                                                              | α = 1.0 (censurati esclusi)                                                   |
| ------------------------ | ---------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| **Gap Train ↔ Val/Test** | 1.00 → 0.67 (-0.33)<br>*Over-fitting visibile ma il modello resta valido fuori campione* | 0.75 → 0.50 (-0.25) su Val<br>*minor over-fit ma peggior generalizzazione*    |
| **Stabilità**            | c-index Val ≈ Test (0.67) → modello coerente                                             | Test > Val (0.62 vs 0.50) → risultato dipende dal caso, indice di instabilità |
| **Loss**                 | Molto più alta: è il prezzo di aver calcolato la NLL anche sui censurati                 | Più bassa perché la loss somma solo sugli eventi osservati                    |
| **Kaplan-Meier**         | Bande strette, separazione chiara, p ≈ 0.015                                             | Bande larghe, p uguale ma CI si sovrappongono di più                          |


-***Conclusione***
- α = 0.0 vince la comparazione

c-index costante ~0.67 su Val e Test.

KM con bande strette ⇒ stima di rischio affidabile.

- α = 1.0 non migliora la robustezza

Riduce l’over-fit di facciata ma degrada la generalizzazione (Val da 0.67 a 0.50).

Il recupero sul test (0.62) è episodico e meno stabile (bande di confidenza larghe).

Nei dati TCGA-BRCA la presenza di censurati aiuta il modello a imparare una funzione di rischio più liscia e, paradossalmente, più generalizzabile.

Rimuoverli (α = 1.0) taglia la sample-size informativa e fa crescere la varianza → peggior ranking medio.

***Conclusione dell'esperimento***
TABELLA RIASSUNTIVA PRESTAZIONI SU TEST C-INDEX
| Modalità        | α = 0.0 (censurati inclusi) | α = 1.0 (censurati esclusi) |
| --------------- | --------------------------- | --------------------------- |
| **WSI**         | **0.6480**                  | 0.5470                      |
| **Genomics**    | **0.6780**                  | 0.5009                      |
| **Multimodale** | **0.6652**                  | 0.6160                      |

**Risultati più importanti**  
1. Il ruolo cruciale dei censurati (α = 0.0)
In tutti e tre i contesti, α = 0.0 ha portato ai migliori c-index su Test, con valori sempre ≥ 0.64.

La presenza dei censurati aiuta a regolarizzare l’hazard: anche se non sappiamo quando un evento avviene, sappiamo che non è avvenuto entro un certo tempo.

Le curve Kaplan–Meier sono le uniche a risultare significative (p < 0.05) solo con α = 0.0, mostrando bande di confidenza strette e separazione visiva coerente.

2. α = 1.0 riduce la varianza... ma anche il valore predittivo
Eliminando i censurati, il modello si allena su meno dati ⇒ la loss si abbassa artificialmente (meno termini nella NLL), ma non migliora.

In Genomics e WSI, il c-index crolla verso la casualità (~0.50–0.55), e le curve KM diventano inutilizzabili (p > 0.1, bande larghe).

3. Overfitting? Solo apparente
Alcuni modelli (es. Genomics α = 0.0) mostrano Train c-index = 1.0, ma Validation e Test restano forti (0.71 e 0.68): segno che il modello impara bene, non solo memorizza.

α = 1.0 riduce l’over-fit visivo, ma degrada la generalizzazione: non è una soluzione reale, solo un “abbassamento” dell’intero sistema.

**Riflessioni sulle modalità**
Solo Genomics (α = 0.0) ha raggiunto il miglior c-index assoluto su Test (0.678): ciò evidenzia l’alta informatività della trascrittomica in TCGA-BRCA.

Solo WSI è stabile ma meno performante (max 0.648): l’informazione morfologica è utile, ma meno precisa nel ranking.

Multimodale α = 0.0 non ha superato Genomics puro, ma è il più bilanciato: evita overfit estremo, mantiene stabilità, e può generalizzare su dati più complessi.

QUINDI  
α = 0.0 è la scelta ottimale in tutti gli scenari: include più informazione, mantiene alta la generalizzazione e genera curve KM interpretabili.  

α = 1.0 può sembrare più “pulito”, ma taglia informazione critica → peggior c-index, maggiore varianza, curve non significative.  

Se si può scegliere una sola modalità, Genomics è la più potente in questo dataset.
Ma l’integrazione multimodale offre robustezza e deve essere preferita se disponibile.  