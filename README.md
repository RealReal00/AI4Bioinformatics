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
  ""task_type": "Survival",
  "max_patches": 64,                  # Patch visive per paziente
  "batch_size": 1,                    # Batch per device
  "real_batch_size": 8,              # Batch effettivo tramite gradient accumulation
  "n_bins": 4,                        # Discretizzazione temporale
  "sample": True,                     # Sampling patch in training
  "test_sample": False,             # Tutte le patch nel test
  "train_size": 0.7, "val_size": 0.15, "test_size": 0.15
  "load_slides_in_RAM": False,
  "num_workers": 0,

  ### Motivazioni
  Il numero ridotto di patch (64) è un compromesso tra tempo di training e capacità predittiva.

  Il batch_size=1 è stato scelto per limitazioni hardware ma compensato da real_batch_size=8 per stabilità.

  La suddivisione bilanciata del dataset è stata curata per evitare set di validazione completamente censurati.

  ### Modello usato: ABMIL_Multimodal
  Il modello impiega un'architettura Multiple Instance Learning con meccanismi di attenzione per le patch visive, integrando in parallelo un ramo per i dati genomici. La combinazione delle due modalità consente al modello di sfruttare indizi sia morfologici che molecolari.

  ### Training e Metriche
  Il modello è stato allenato per 10 epoche. I risultati evidenziano una curva di apprendimento stabile e una crescente capacità del modello nel predire la sopravvivenza:
  Split	          Miglior c-index	       Loss

  Train	          0.9597	              0.215
  Validation	    0.5364	              1.598
  Test	          0.6575	              2.106

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
  ""task_type": "Survival",
  "max_patches": 4096,                  # Patch visive per paziente
  "batch_size": 1,                    # Batch per device
  "real_batch_size": 8,              # Batch effettivo tramite gradient accumulation
  "n_bins": 4,                        # Discretizzazione temporale
  "sample": True,                     # Sampling patch in training
  "test_sample": False,             # Tutte le patch nel test
  "train_size": 0.7, "val_size": 0.15, "test_size": 0.15
  "load_slides_in_RAM": False,
  "num_workers": 0

  ### Motivazioni
  L’aumento di max_patches da 64 a 4096 è pensato per dare al modello una visione più completa del tessuto tumorale, migliorando la rappresentazione istopatologica. Tuttavia, per evitare problemi di memoria, il caricamento in RAM è disattivato e si mantiene num_workers = 1 per sicurezza, a scapito delle prestazioni in tempo.

  ### Prestazione del modello
  Split	    Miglior c-index	      Loss

  Train	        0.6605	          0.4472
  Val	          0.6585	          0.5892
  Test	        0.6790	          1.5790

  Il c-index vicino a 0.66–0.68 su tutti i dataset mostra che il modello ha acquisito una capacità predittiva significativa, ben superiore al random (0.5).

  Il loss sul test è più alto, ma coerente con la maggiore complessità dei dati da generalizzare (tutte le patch incluse).

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

