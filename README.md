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


  ## Esperimento 1: Modello ABMIL_Multimodal per Sopravvivenza
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
  Split	                    c-index	       Loss

  Train	                    0.9597	         0.215
  Validation	              0.5364	         1.598
  Test(best model)          0.6449           1.117
  Test (last epoch)	        0.6575	         2.106

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

## Esperimento 2: Modello ABMIL_Multimodal per Sopravvivenza

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
  Split	                  c-index	          Loss

  Train	                  0.6605	          0.4472
  Validation	            0.6585	          0.5893
  Test (best model)     	0.6652	          1.6387
  Test (last epoch)	      0.6838	          1.5763

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

  ## ESPERIMENTO INFLUENZA DELLA CENSURA CON DIVERSI N_BINS E ALPHA
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

Best model vs last epoch
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


