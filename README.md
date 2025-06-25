  ## ESPERIMENTO SULLA INFLUENZA DEL CAMBIAMENTO DEL NUMERO DI PATCH

---

### **64 PATCH**

Lo scopo dell'esperimento è valutare l'efficacia del modello ABMIL_Multimodal, una rete neurale capace di combinare dati visivi (immagini istopatologiche WSI) e genomici, per prevedere la sopravvivenza dei pazienti affetti da carcinoma mammario (TCGA-BRCA).

Il problema è affrontato come un task di sopravvivenza (Survival Analysis), con dati parzialmente censurati. In particolare, l’obiettivo è apprendere una curva di sopravvivenza per ogni paziente, e stimare con accuratezza il rischio.

---

#### Configurazione degli iperparametri usati

| Parametro            | Valore     | Descrizione                               |
|----------------------|------------|-------------------------------------------|
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

---

#### Motivazioni

Nel task di survival prediction basato su immagini WSI, l’informazione visiva viene frammentata in patch. Tuttavia, non è chiaro quante patch siano davvero necessarie per una predizione efficace.

In questo esperimento si confrontano due scenari estremi e infine un compromesso tra i due:

- **64 patch**: configurazione ridotta, focalizzata, potenzialmente più rapida e meno rumorosa  
- **4096 patch**: uso estensivo dell’informazione visiva disponibile, ma con alto costo computazionale e rischio di rumore  
- **512 patch**: permette di coprire una buona porzione della WSI pur mantenendo una complessità gestibile

Verificare se un numero elevato di patch (4096/512) porta realmente a un miglioramento delle metriche predittive rispetto a una selezione contenuta (64).  
Analizzare l’impatto sul c-index, sulla stabilità delle curve KM e sul rischio di overfitting.

---

#### Training e Metriche

Il modello è stato allenato per 10 epoche. I risultati evidenziano una curva di apprendimento stabile e una crescente capacità del modello nel predire la sopravvivenza:

| Split             | C-index | Loss  |
|------------------|---------|-------|
| Train            | 1.0000  | 0.001 |
| Validation       | 0.6631  | 1.437 |
| Test (best model)| 0.6937  | 1.567 |
| Test (last epoch)| 0.6937  | 1.567 |

Il c-index su validation/test resta moderato, ma significativamente migliore del caso random (0.5), soprattutto sul test set.  
Il gap tra training e test suggerisce overfitting: il modello ha appreso bene il training set, ma la generalizzazione è limitata.

---

- **Punti Positivi**  
  Train performance molto alta (C-index = 1): il modello ABMIL ha appreso una forte distinzione nei dati di addestramento, anche se probabilmente sono memorizzati piuttosto che imparati.  
  Dal test notiamo un C-index alto quindi una probabile capacità di generalizzare ma il gap col training conferma l'overfitting.

- **Punti Critici**  
  La loss del modello sia all’ultima epoca sia al best model sono significativamente alte (1.567).  
  Log-rank p-values nei Kaplan-Meier plot p = 0.1253: dimostrano che i gruppi ad alto e basso rischio hanno differenze statisticamente irrilevanti nella sopravvivenza.

---

**Osservazioni**  
L'uso di sole 64 patch non è insufficiente per catturare la complessità morfologica del tessuto, soprattutto nei WSI ad alta eterogeneità.

---

### **4096 PATCH**

#### Configurazione degli iperparametri usati

| Parametro            | Valore     | Descrizione                               |
|----------------------|------------|-------------------------------------------|
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

---

#### Motivazioni

L’aumento di `max_patches` da 64 a 4096 è pensato per dare al modello una visione più completa del tessuto tumorale, migliorando la rappresentazione istopatologica.  
Tuttavia, per evitare problemi di memoria, il caricamento in RAM è disattivato e si mantiene `num_workers = 0` per sicurezza, a scapito delle prestazioni in tempo.

---

#### Prestazione del modello

| Split             | C-index | Loss   |
|------------------|---------|--------|
| Train            | 1.0000  | 0.0025 |
| Validation       | 0.6722  | 1.5348 |
| Test (best model)| 0.6652  | 1.6387 |
| Test (last epoch)| 0.6838  | 1.5763 |

Il c-index vicino a 0.66–0.68 su tutti i dataset mostra che il modello ha acquisito una capacità predittiva significativa, ben superiore al random (0.5).  
La loss sul test è più alta, ma coerente con la maggiore complessità dei dati da generalizzare (tutte le patch incluse).

---

### Generalizzazione

- Il **miglior modello con 64 patch** ottiene il **c-index test più alto**: **0.694**
- Con **4096 patch**, il `best-epoch` è **più basso (0.665)**  
  ma **l’ultima epoca** risale a **0.684**  
  → Il modello non era ancora arrivato a convergenza

---

### Stabilità

- Differenza **train–validation** ampia (~0.33) in entrambi i casi → **overfitting evidente**
- Con **4096 patch**, il gap si riduce leggermente:
  - `1 – 0.672 = 0.328` vs `1 – 0.663 = 0.337`
- Ma la riduzione **non è sostanziale**

---

### Rumore vs Informazione

- Con **64 patch**:
  - Si campionano **regioni più grandi**
  - Meno patch → **meno rumore di fondo**
  - Ma anche **rischio di perdere** aree rilevanti

- Con **4096 patch**:
  - Si cattura **molta più eterogeneità**
  - Ma si include anche **molto background** non informativo
  - Il modello deve **imparare a ignorare il rumore**
  - ⇒ leggero **peggioramento delle performance** su test

---

### Aspetto visivo delle curve Kaplan–Meier

**64 patch**

- Early-phase (0–3 anni): curve **abbastanza vicine** → difficile distinguere eventi precoci
- Mid/Late-phase (5–10 anni): divergenza evidente
  - **High risk** scende sotto 0.60
  - **Low risk** resta ≈ 0.75–0.80
- Gli intervalli di confidenza iniziano a separarsi attorno al **7°–8° anno**

**4096 patch**

- Curve appaiate su tutta la finestra temporale
- Vantaggio Low risk **non sistematico**
- Intervalli di confidenza **quasi totalmente sovrapposti**

---

### Conclusione

Si è voluto testare il modello che potesse rappresentare un buon compromesso tra efficienza computazionale e capacità predittiva.

L’ipotesi è che:
- **64 patch** siano troppo poche → rischio di trascurare regioni tumorali importanti
- **4096 patch** introducano troppo rumore → peggioramento delle performance
- **512 patch** offrano un buon compromesso tra copertura e controllo del rumore

---

## CONFRONTO FINALE – Risultati numerici (C-index e Loss)

| Split          | 64 patch        | 512 patch         | 4096 patch        |
|----------------|------------------|-------------------|-------------------|
| Train (best)   | 1.0000 / 0.001    | 0.8474 / 0.233     | 1.0000 / 0.0025    |
| Validation     | 0.6631 / 1.437    | 0.6697 / 0.588     | 0.6722 / 1.5348    |
| Test (best)    | 0.6937 / 1.567    | **0.7010 / 0.668** | 0.6652 / 1.6387    |
| Overfitting    | **Alto**         | **Moderato**       | **Alto**           |

---

### Il modello con **512 patch** ottiene:

- Il **C-index test più alto**: **0.701**
- Una **loss molto più bassa**, sia su **validation** che **test**
- Una **generalizzazione più stabile**:
  - Train C-index non perfetto (0.847)
  - → Il modello non ha overfittato brutalmente

---

## Curva di Kaplan–Meier (512 patch)

| Aspetto         | Osservazione                                                               |
|------------------|----------------------------------------------------------------------------|
| Separazione      | **Netta** tra Low e High risk (curve ben distinte)                        |
| Andamento        | High risk **scende progressivamente**, Low risk si **stabilizza**         |
| p-value          | **0.0035** → altamente significativo (**p < 0.01**)                        |
| Intervalli CI    | Ben separati dalla metà in poi, **sovrapposizione minima**                |

> Questo è il **miglior risultato visivo e statistico** tra tutti i test:  
> il modello ha appreso una **distinzione prognostica chiara e robusta**.




  ###  Configurazione della Loss Function nei due precedenti esperimenti e Considerazioni sulla Censura
   - alpha=0.0  
   - eps=1e-7  
   - reduction='mean'  

   Quando si considerano sia censurati che non si permette al modello di apprendere anche da pazienti con eventi non osservati. Questo è cruciale in contesti reali come il carcinoma mammario (TCGA-BRCA), dove una porzione significativa dei pazienti è censurata ( infatti nel dataset abbiamo 912 censurati e 146 non censurati).

   Tuttavia, includere i censurati comporta anche una riduzione del segnale diretto sull'evento (morte/recidiva), che può influenzare la precisione nei confronti di chi ha avuto effettivamente l'evento

## ESPERIMENTO SULLA INFLUENZA DELLA CENSURA CON DIVERSI N\_BINS E ALPHA

Il problema è ancora una volta affrontato come un task di sopravvivenza (Survival Analysis) usando **ABMIL\_Multimodal** con input `["WSI", "Genomics"]`, ma ora si vogliono confrontare i risultati e le metriche per ogni combinazione `alpha`–`n_bins`.
In particolare, l’obiettivo è apprendere una curva di sopravvivenza per ogni paziente e stimare con accuratezza il rischio.

I test sono stati condotti mettendo a confronto le combinazioni generate dai valori di **α (0.0 e 1.0)** e i **bin (2 e 4)**.

Per ciascun test, sono stati salvati:

* I modelli (`.pt`) migliori (best val loss)
* Metriche su train, val, test per best model
* Last epoch e curve di Kaplan‑Meier per i test set

---

## Analisi: Effetto di α con 2 bin

### Tabella – α = 0.0, bin = 2

| Alpha | Bins | Split | **C‑index** | **Loss (mean)** | Tipo modello |
| :---: | :--: | :---: | :---------: | :-------------: | :----------: |
|  0.0  |   2  | Train |  **1.000**  |      0.002      |  Best Model  |
|  0.0  |   2  |  Val  |    0.696    |      1.365      |  Best Model  |
|  0.0  |   2  |  Test |  **0.771**  |      1.478      |  Best Model  |
|  0.0  |   2  | Train |    1.000    |      0.002      |  Last Epoch  |
|  0.0  |   2  |  Val  |    0.696    |      1.365      |  Last Epoch  |
|  0.0  |   2  |  Test |    0.771    |      1.478      |  Last Epoch  |

### Tabella – α = 1.0, bin = 2

| Alpha | Bins | Split | **C‑index** | **Loss (mean)** | Tipo modello |
| :---: | :--: | :---: | :---------: | :-------------: | :----------: |
|  1.0  |   2  | Train |    0.702    |      0.058      |  Best Model  |
|  1.0  |   2  |  Val  |  **0.411**  |    **0.313**    |  Best Model  |
|  1.0  |   2  |  Test |    0.610    |      0.330      |  Best Model  |
|  1.0  |   2  | Train |    0.702    |      0.058      |  Last Epoch  |
|  1.0  |   2  |  Val  |    0.411    |      0.313      |  Last Epoch  |
|  1.0  |   2  |  Test |    0.610    |      0.330      |  Last Epoch  |

## Osservazioni

### Effetto di α (alpha)

* Con **α = 0.0**, il modello include anche i pazienti censurati.

  * **C‑index** → Train: `1.000`, Val: `0.696`, Test: `0.771`.
  * La **perdita (NLL media)** è quasi nulla sul train (`0.002`) e sale a `1.48` sul test: il modello **over‑fit sul ranking**, ma riesce comunque a trasferire parte della struttura di rischio fuori campione.

* Con **α = 1.0** (esclusione dei censurati), il modello dispone di **meno esempi informativi** e fatica a generalizzare:

  * **C‑index** → Train: `0.702`, Val: `0.411`, Test: `0.610`.
  * La validazione è quasi casuale, segno che **senza l’informazione censurata il modello non impara un hazard affidabile**.

> **Conclusione:** con solo 2 bin temporali, il contributo dei censurati è **fondamentale** per ottenere un ranking robusto.

### Comportamento del Best Model vs Last Epoch

* In entrambi i casi **best** e **last** coincidono: l’early stopping si attiva non appena la val‑loss smette di scendere.
* Per **α = 0.0** questo evita che l’overfit peggiori ulteriormente; per **α = 1.0** l’allenamento si interrompe presto perché la val‑loss non migliora mai.

### Curve di Kaplan–Meier

|  α  | Checkpoint  | Log‑rank *p* | Lettura qualitativa                                                                                                                                     |
| :-: | :---------- | :----------: | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 0.0 | Best / Last |  **0.0053**  | Il gruppo **High‑risk** (blu) crolla entro 10 anni, mentre **Low‑risk** (arancio) mantiene >80 % di sopravvivenza. Separazione altamente significativa. |
| 1.0 | Best / Last |  **0.0150**  | Curve ancora separate (*p* < 0.05), ma le bande di confidenza si sovrappongono di più rispetto a α = 0.0.                                               |


---

## Analisi: Effetto di α con 4 bin

### Tabella – α = 0.0, bin = 4

| Alpha | Bins | Split | **C‑index** | **Loss (mean)** | Tipo modello |
| :---: | :--: | :---: | :---------: | :-------------: | :----------: |
|  0.0  |   4  | Train |    0.9988   |      0.0066     |  Best Model  |
|  0.0  |   4  |  Val  |    0.6692   |      1.2141     |  Best Model  |
|  0.0  |   4  |  Test |    0.6851   |      1.2816     |  Best Model  |
|  0.0  |   4  | Train |    0.9988   |      0.0066     |  Last Epoch  |
|  0.0  |   4  |  Val  |    0.6692   |      1.2141     |  Last Epoch  |
|  0.0  |   4  |  Test |    0.6851   |      1.2816     |  Last Epoch  |

### Tabella – α = 1.0, bin = 4

| Alpha | Bins | Split | **C‑index** | **Loss (mean)** | Tipo modello |
| :---: | :--: | :---: | :---------: | :-------------: | :----------: |
|  1.0  |   4  | Train |    0.7542   |      0.0229     |  Best Model  |
|  1.0  |   4  |  Val  |    0.4987   |      0.3352     |  Best Model  |
|  1.0  |   4  |  Test |    0.6165   |      0.4615     |  Best Model  |
|  1.0  |   4  | Train |    0.7543   |      0.0229     |  Last Epoch  |
|  1.0  |   4  |  Val  |    0.4982   |      0.3350     |  Last Epoch  |
|  1.0  |   4  |  Test |    0.6160   |      0.4615     |  Last Epoch  |

## Osservazioni

### Effetto di α

* **α = 0.0** domina nettamente: +0.07 di C‑index su test (`0.685 vs 0.616`); loss più alta perché include censurati, ma **più clinicamente realistica**.
* **α = 1.0** riduce la varianza: C‑index in validazione ≈ 0.50 → **ranking casuale**.

### Comportamento Best vs Last

* Con **α = 0.0**: curva di validazione piatta → early‑stopping **non migliora** l’ultima epoca.
* Con **α = 1.0**: best/last **quasi identici** → il modello **non riesce a generalizzare**
### Performance generale

* **α = 0.0**: C‑index test ≈ 0.69.
* **α = 1.0**: test < 0.62, validazione ≈ 0.50.

### Curve di Kaplan–Meier

|  α  | Checkpoint | Log‑rank *p* | Commento rapido                                                   |
| :-: | :--------- | :----------: | ----------------------------------------------------------------- |
| 0.0 | best/last  |   `0.0040`   | Bande ben distinte, separazione forte.                            |
| 1.0 | best/last  |   `0.0157`   | Separazione ancora significativa, ma bande **molto sovrapposte**. |


---

## Conclusione finale (bin = 4)

* **α = 0.0 + 4 bin** resta il setup migliore: miglior C‑index complessivo, curve KM ben separate (*p* < 0.005) e overfit sul train **non danneggia** val/test.
* **α = 1.0** riduce la complessità, ma **perde informazione critica**: ranking in validazione casuale e incertezza più alta.

> In presenza di molti pazienti censurati, includerli (**α = 0.0**) è **cruciale**. Con 4 bin si ottiene un buon compromesso tra **dettaglio temporale e potere statistico**.





## ESPERIMENTO SULLA INFLUENZA DELLA DIVERSE MODALITA' CON DIVERSI ALPHA

* **Obiettivo**
  L’obiettivo di questo esperimento è analizzare come le diverse modalità di input (Genomics, WSI e Multimodale) rispondono alla presenza o assenza della censura nel task di Survival Prediction.
  L'effetto di α potrebbe dipendere dalla modalità di input, questo perché:
* la genomica potrebbe contenere segnali più deboli e più sensibili al rumore;
* le immagini WSI potrebbero beneficiare della presenza di più esempi, anche censurati;
* la modalità multimodale potrebbe compensare le debolezze delle singole fonti.

---

### Caso **solo Genomics**

* **α = 0.0 – Genomics – bin = 4**

| Alpha | Bins | Split |    C-index |   Loss | Tipo modello |
| ----- | ---- | ----- | ---------: | -----: | ------------ |
| 0.0   | 4    | Train | **1.0000** | 0.0032 | Best Model   |
| 0.0   | 4    | Train |     1.0000 | 0.0032 | Last Epoch   |
| 0.0   | 4    | Val   | **0.7093** | 1.5820 | Best Model   |
| 0.0   | 4    | Val   |     0.7093 | 1.5820 | Last Epoch   |
| 0.0   | 4    | Test  | **0.6780** | 1.8335 | Best Model   |
| 0.0   | 4    | Test  |     0.6780 | 1.8335 | Last Epoch   |

**Kaplan–Meier (log‑rank p ≈ 0.028)**
Il gruppo High‑risk (blu) mostra una discesa rapida: la sopravvivenza si riduce sotto il 40 % entro \~7 anni, mentre il gruppo Low‑risk (arancione) si mantiene stabilmente sopra il 60 %.
Le bande di confidenza restano ben separate fino all’ottavo anno; il p‑value < 0.05 segnala una separazione statisticamente significativa, coerente con un c‑index ≈ 0.68.  


* **α = 1.0 – Genomics – bin = 4**

| Alpha | Bins | Split |    C-index |   Loss | Tipo modello |
| ----- | ---- | ----- | ---------: | -----: | ------------ |
| 1.0   | 4    | Train | **0.7711** | 0.0074 | Best Model   |
| 1.0   | 4    | Train |     0.7711 | 0.0074 | Last Epoch   |
| 1.0   | 4    | Val   | **0.4921** | 0.5443 | Best Model   |
| 1.0   | 4    | Val   |     0.4921 | 0.5443 | Last Epoch   |
| 1.0   | 4    | Test  | **0.5009** | 0.7345 | Best Model   |
| 1.0   | 4    | Test  |     0.5009 | 0.7345 | Last Epoch   |  

**Kaplan–Meier (log‑rank p ≈ 0.414)**
Le curve High‑ e Low‑risk restano vicine per gran parte del tempo, con bande di confidenza ampie che si sovrappongono.
Il p‑value > 0.40 indica assenza di separazione significativa. Il c‑index prossimo a 0.50 conferma un modello incapace di distinguere i gruppi.  


#### Osservazioni Genomics

| Aspetto                  | α = 0.0 (censurati inclusi)                                                               | α = 1.0 (censurati esclusi)                                                       |
| ------------------------ | ----------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| **Gap Train ↔ Val/Test** | 1.00 → 0.71 / 0.68 (Δ ≈ –0.32)<br>*over‑fit evidente ma buone prestazioni fuori campione* | 0.77 → 0.49 / 0.50 (Δ ≈ –0.27)<br>*prestazioni scarse anche su validation e test* |
| **Stabilità**            | c‑index Val ≈ Test (0.709 ↔ 0.678) ⇒ generalizzazione stabile                             | c‑index Val ≈ Test (0.492 ↔ 0.501) ⇒ modello vicino al ranking casuale            |
| **Loss**                 | Train 0.003 → Test 1.834: la NLL è mediata                 | Train 0.007 → Test 0.735: loss più bassa ma non porta beneficio                   |
| **Kaplan‑Meier**         | Bande strette, p ≈ 0.028 ⇒ separazione significativa e utile clinicamente                 | Bande ampie, p ≈ 0.41 ⇒ nessuna evidenza di separazione tra i gruppi              |

**Conclusione Genomics**
α = 0.0 resta chiaramente vincente:

* C‑index elevato e stabile su validation (0.709) e test (0.678).
* Curve Kaplan–Meier ben separate e significative (p < 0.05), a supporto della capacità discriminativa del modello.
  α = 1.0 riduce l’over‑fit di facciata ma abbatte la generalizzazione (val/test ≈ 0.50).

---

### Caso **solo WSI**

* **α = 0.0 – WSI – bin = 4**

| Alpha | Bins | Split |    C-index |   Loss | Tipo modello |
| ----- | ---- | ----- | ---------: | -----: | ------------ |
| 0.0   | 4    | Train | **0.8903** | 0.2082 | Best Model   |
| 0.0   | 4    | Train |     0.8903 | 0.2082 | Last Epoch   |
| 0.0   | 4    | Val   | **0.6403** | 0.6457 | Best Model   |
| 0.0   | 4    | Val   |     0.6403 | 0.6457 | Last Epoch   |
| 0.0   | 4    | Test  | **0.6488** | 0.7164 | Best Model   |
| 0.0   | 4    | Test  |     0.6488 | 0.7164 | Last Epoch   |

**Kaplan–Meier (log‑rank p ≈ 0.090)**
Il gruppo High‑risk (arancione) scende più rapidamente: la sopravvivenza si dimezza intorno al 7‑8° anno, mentre il gruppo Low‑risk (blu) resta sopra il 60 %.
Il p‑value ≈ 0.09 non raggiunge la soglia di significatività ma indica una tendenza coerente con il c‑index ≈ 0.65.  


* **α = 1.0 – WSI – bin = 4**

| Alpha | Bins | Split |    C-index |   Loss | Tipo modello |
| ----- | ---- | ----- | ---------: | -----: | ------------ |
| 1.0   | 4    | Train | **0.5797** | 0.1651 | Best Model   |
| 1.0   | 4    | Train |     0.5797 | 0.1651 | Last Epoch   |
| 1.0   | 4    | Val   | **0.5464** | 0.1940 | Best Model   |
| 1.0   | 4    | Val   |     0.5464 | 0.1940 | Last Epoch   |
| 1.0   | 4    | Test  | **0.5470** | 0.2431 | Best Model   |
| 1.0   | 4    | Test  |     0.5470 | 0.2431 | Last Epoch   |

**Kaplan–Meier (log‑rank p ≈ 0.106)**
La separazione tra le curve è più debole: le bande di confidenza si sovrappongono quasi interamente fin dall’inizio.
Il p‑value > 0.10 e il c‑index ≈ 0.55 confermano che il ranking è vicino al caso.  


#### Osservazioni WSI

| Aspetto                  | α = 0.0 (censurati inclusi)                                                      | α = 1.0 (censurati esclusi)                                             |
| ------------------------ | -------------------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| **Gap Train ↔ Val/Test** | 0.890 → 0.640 (Δ ≈ –0.25)<br>*over‑fit visibile ma ranking buono fuori campione* | 0.580 → 0.547 (Δ ≈ –0.03)<br>*poco over‑fit, ma livello vicino al caso* |
| **Stabilità**            | c‑index Val ≈ Test (0.640 ↔ 0.649) ⇒ comportamento coerente                      | c‑index Val ≈ Test (0.546 ↔ 0.547) ⇒ stabile ma poco informativo        |
| **Loss**                 | Train 0.208 → Test 0.716: loss più alta su test perché include censurati         | Train 0.165 → Test 0.243: loss più bassa ma non migliora il ranking     |
| **Kaplan‑Meier**         | Bande relativamente strette, trend di separazione (p ≈ 0.09)                     | Bande ampie, sovrapposizione forte, p ≈ 0.11                            |

**Conclusione WSI**
α = 0.0 è preferibile: c‑index \~ 0.65 stabile e migliore separazione delle curve.
α = 1.0 non porta reale robustezza: riduce over‑fit ma il ranking resta vicino al caso.

---

### Caso **WSI + Genomics (Multimodale)**

* **α = 0.0 – WSI + Genomics – bin = 4**

| Alpha | Bins | Split |    C-index |   Loss | Tipo modello |
| ----- | ---- | ----- | ---------: | -----: | ------------ |
| 0.0   | 4    | Train | **0.9988** | 0.0066 | Best Model   |
| 0.0   | 4    | Val   | **0.6692** | 1.2141 | Best Model   |
| 0.0   | 4    | Test  | **0.6851** | 1.2816 | Best Model   |
| 0.0   | 4    | Train |     0.9988 | 0.0066 | Last Epoch   |
| 0.0   | 4    | Val   |     0.6692 | 1.2141 | Last Epoch   |
| 0.0   | 4    | Test  |     0.6851 | 1.2816 | Last Epoch   |

**Kaplan–Meier (log‑rank p ≈ 0.0150)**
La curva High‑risk (blu) precipita rapidamente entro \~8 anni, mentre la Low‑risk (arancione) scende più lentamente; le bande di confidenza non si sovrappongono dopo il 5º anno. Il p‑value < 0.02 conferma una buona stratificazione. L'over‑fit è visibile sul train (c‑index ≈ 1), ma la val/test restano solidi (\~0.67‑0.69).  
>

* **α = 1.0 – WSI + Genomics – bin = 4**

| Alpha | Bins | Split |    C-index |   Loss | Tipo modello |
| ----- | ---- | ----- | ---------: | -----: | ------------ |
| 1.0   | 4    | Train | **0.7542** | 0.0229 | Best Model   |
| 1.0   | 4    | Val   | **0.4987** | 0.3352 | Best Model   |
| 1.0   | 4    | Test  | **0.6165** | 0.4615 | Best Model   |
| 1.0   | 4    | Train |     0.7542 | 0.0229 | Last Epoch   |
| 1.0   | 4    | Val   |     0.4987 | 0.3352 | Last Epoch   |
| 1.0   | 4    | Test  |     0.6165 | 0.4615 | Last Epoch   |

**Kaplan–Meier (log‑rank p ≈ 0.0157)**
Eliminando i censurati le bande di confidenza si allargano, specie nel gruppo Low‑risk; la separazione resta visibile e il p‑value rimane significativo. Tuttavia la val c‑index crolla a \~0.50, segno di bassa generalizzazione, mentre il test recupera parzialmente (0.616). 


#### Osservazioni Multimodale

| Aspetto                  | α = 0.0 (censurati inclusi)                                                         | α = 1.0 (censurati esclusi)                                               |
| ------------------------ | ----------------------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| **Gap Train ↔ Val/Test** | 1.00 → 0.67/0.69 (Δ ≈ –0.33) – over‑fit evidente ma buone prestazioni out‑of‑sample | 0.75 → 0.50/0.62 (Δ ≈ –0.25) – minor over‑fit ma peggior generalizzazione |
| **Stabilità**            | c‑index Val ≈ Test (0.669 ↔ 0.685) – modello coerente                               | Test > Val (0.616 > 0.499) – risultato instabile                          |
| **Loss**                 | Più alta (1.28 su test)                                  | Più bassa (0.46 su test) ma meno informativa                              |
| **Kaplan‑Meier**         | Bande strette, separazione chiara, p ≈ 0.015                                        | Bande più larghe, separazione ancora significativa ma meno netta          |

**Conclusione Multimodale**
α = 0.0 vince: c‑index costante \~ 0.67‑0.69, KM con bande strette e p‑value < 0.02.
α = 1.0 riduce il gap train‑val ma degrada la generalizzazione: val c‑index scende a \~0.50 e test è meno stabile.

---

### Tabella Riassuntiva (C‑index Test)

| Modalità        |    α = 0.0 | α = 1.0 |
| --------------- | ---------: | ------: |
| **Genomics**    | **0.6780** |  0.5009 |
| **WSI**         | **0.6488** |  0.5470 |
| **Multimodale** | **0.6851** |  0.6165 |

### Risultati Essenziali

1. **Censurati fondamentali (α = 0.0)**
   In tutte le modalità l’inclusione dei censurati produce i c‑index più alti (≥ 0.64) e curve KM statisticamente rilevanti.

2. **α = 1.0 riduce la varianza ma abbassa il potere predittivo**
   Il taglio dei censurati fa crollare il c‑index (Genomics/WSI) e allarga le bande di confidenza.

3. **Over‑fitting solo apparente**
   C‑index = 1.0 sul train non si traduce in degrado su val/test quando α = 0.0.

4. **Confronto Modalità**

   * Genomics α = 0.0 ottiene il miglior c‑index assoluto (0.678).
   * WSI è stabile ma leggermente inferiore (0.649).
   * Multimodale α = 0.0 è il più bilanciato, combinando stabilità e robustezza.

**Conclusione**
α = 0.0 è la scelta ottimale in tutti gli scenari: sfrutta l’informazione dei censurati, massimizza la generalizzazione e genera curve KM interpretabili.
Se si sceglie una sola fonte, Genomics è la più potente, ma l’integrazione multimodale offre robustezza extra.  

---


## Esperimento MCAT-Lite (α = 0, bin = 4 – WSI + Genomics)

### Motivazioni dell’esperimento

Il setup **WSI + Genomics** con **α = 0.0 / bin = 4** era il migliore fra i modelli ABMIL (c-index ≈ 0.685).

Lo scopo è verificare se l’architettura **MCAT-Lite**, pur semplificata, riesca a sfruttare meglio le interazioni fra patch istologiche e profili trascrittomici, migliorando il ranking di sopravvivenza rispetto a un ABMIL “a concatenazione”.

---

### Iper-parametri

| Parametro           | Valore | Descrizione                          |
|---------------------|--------|--------------------------------------|
| input_dim           | 1024   | Feature patch WSI                    |
| genomics_input_dim  | 19962  | Geni RNA-seq                         |
| hidden_dim          | 256    | Dimensione spazio latente del Transformer |
| n_heads             | 8      | Teste multi-head                     |
| n_layers            | 2      | Layer encoder                        |
| dropout             | 0.1    | Probabilità dropout                  |
| output_dim          | 4      | Bin temporali                        |
| max_patches         | 4096   | Patch WSI massimo per paziente       |

---

### Risultati (Loss = Mean)

| Split  | Checkpoint | c-index | NLL-loss (mean) |
|--------|------------|---------|-----------------|
| Train  | best       | 0.7258  | 0.3749          |
| Val    | best       | 0.6545  | 0.6101          |
| Test   | best       | 0.5358  | 0.7803          |


---

### Osservazioni

- **Generalizzazione**: il c-index scende di ~0.07 dal train al validation e di un ulteriore ~0.12 sul test (−0.19 complessivo), segnalando un moderato over-fit e limitata tenuta fuori campione.
- **Prestazioni assolute**: 0.5358 è inferiore sia all’ABMIL multimodale (0.6851) sia al Genomics-only (0.6780). Il Transformer compatto non migliora le prestazioni complessive.
- **Kaplan–Meier**: le curve dei gruppi di rischio si sovrappongono e il log-rank è p ≈ 0.60 ⇒ nessuna separazione statisticamente significativa.



---

### Confronto modelli

| Modello     | Modalità         | c-index Test |
|-------------|------------------|--------------|
| ABMIL       | Genomics         | 0.6780       |
| ABMIL       | WSI              | 0.6488       |
| ABMIL       | WSI + Genomics   | 0.6851       |
| MCAT-Lite   | WSI + Genomics   | 0.5358       |

---

### Conclusione

In questo setup **MCAT-Lite non migliora il ranking di sopravvivenza**:  
- Il c-index peggiora rispetto a tutti i baseline ABMIL.  
- La stratificazione KM non è significativa.

**Probabile causa**:
- Assenza dei blocchi di co-attention propri del MCAT completo → interazione patch-gene poco espressiva.




## Esperimento MCAT-MultimodalTopK (α = 0, bin = 4 – WSI + Genomics)

### Motivazione

Nel benchmark aggiornato **MCAT-Lite non ha migliorato i baseline ABMIL**:  
sul test ha ottenuto un **c-index di 0.5358**, inferiore sia all’**ABMIL WSI + Genomics** (0.6851) che al **Genomics-only** (0.6780).

Il presente esperimento verifica se il modello **MCAT-MultimodalTopK**, con maggiore capacità e selezione delle patch più informative, riesca a colmare il gap lasciato da MCAT-Lite e avvicinarsi (o superare) le prestazioni di ABMIL multimodale.

### Caratteristiche principali

- `hidden_dim = 384` → maggiore capacità
- `n_layers = 3` → Transformer più profondo
- **Top-K pooling**: 512 patch più rilevanti selezionate
- **Self-attention piena** (Linformer disattivato)

> L’idea è aumentare la profondità e filtrare il rumore istologico senza far esplodere i parametri.

---

### Iper-parametri

| Parametro           | Valore | Nota                       |
|---------------------|--------|----------------------------|
| input_dim           | 1024   | embedding patch            |
| genomics_input_dim  | 19962  | geni TPM-log               |
| hidden_dim          | 384    | spazio latente Transformer |
| n_heads             | 8      | self-attention             |
| n_layers            | 3      | encoder condivisi          |
| dropout             | 0.10   | feed-forward dropout       |
| output_dim          | 4      | bin temporali              |
| max_patches         | 4096   | patch campionate           |
| top_k               | 512    | patch tenute dopo scoring  |
| k (Linformer)       | —      | full attention             |

---

### Risultati (Loss = Mean)

| Split  | Checkpoint | c-index | NLL-loss (mean) |
|--------|------------|---------|-----------------|
| Train  | best       | 0.9369  | 0.1704          |
| Val    | best       | 0.6865  | 1.3502          |
| Test   | best       | 0.6618  | 1.3099          |

> *Il checkpoint `best` coincide con l’ultima epoca – training stabile.*

---

### Osservazioni

- **Overfitting marcato**: il c-index passa da **0.937 → 0.687 → 0.662** (train → val → test).  
  ⇒ La maggiore capacità memorizza bene il training ma generalizza solo parzialmente.

- **Confronto con MCAT-Lite**:  
  **+0.126 c-index sul test** (0.662 vs 0.536).  
  Il Top-K riduce nettamente il gap con i baseline e **supera di molto la versione Lite**.

- **Confronto con ABMIL WSI + Genomics**:  
  Rimane un **divario di −0.023 c-index** (0.662 vs 0.685), da colmare probabilmente con regolarizzazione o fusion più stretta fra modalità.

- **Stabilità fra split**:  
  Val-test differenza ≈ 0.025 → miglior coerenza rispetto a run precedenti.

- **Kaplan–Meier**:  
  `log-rank p ≈ 0.017 (< 0.05)` → separazione **statisticamente significativa**,  
  mentre MCAT-Lite era **non significativa (p ≈ 0.60)**.

---

### Interpretazione rapida

| Aspetto         | MCAT-Lite | MCAT-TopK | Commento               |
|-----------------|-----------|-----------|-------------------------|
| hidden_dim      | 256       | 384       | più capacità            |
| layer encoder   | 2         | 3         | profondità extra        |
| Top-K patch     | —         | 512       | patch informative       |
| c-index Test    | 0.5358    | 0.6618    | **+0.126 vs Lite**      |
| Log-rank p      | ~0.60     | 0.017     | **non sig. → sig.**     |

---

### Conclusione

Il modello **MCAT-MultimodalTopK**:

- Migliora sensibilmente MCAT-Lite sia in **ranking** (−overfitting, +0.126 c-index)  
  sia in **significatività KM** (log-rank da non significativo a p < 0.05)
  
- **Non supera ancora ABMIL multimodale**, ma **riduce il gap a soli 0.023 pt**


## ESPERIMENTO ANALISI DELLA VARIABILIA' DEL SEED
*(α = 0, bin = 4 · WSI + Genomics)*

### Scopo

Quantificare quanto la sola inizializzazione casuale influenzi le prestazioni del modello **ABMIL multimodale**.  
Sono state eseguite **5 run identiche** con seed: `0`, `1`, `7`, `42`, `1234`.

---

### Risultati (checkpoint best, Loss = mean)

| Seed  | c-index Train | c-index Val | c-index Test | Loss Train | Loss Val | Loss Test |
|-------|---------------|-------------|--------------|------------|----------|-----------|
| 0     | 0.9694        | 0.7998      | 0.5883       | 0.0424     | 0.6217   | 1.3949    |
| 1     | 0.9952        | 0.6516      | 0.4865       | 0.0168     | 1.0352   | 0.9431    |
| 7     | 0.9842        | 0.7710      | 0.6053       | 0.0722     | 0.8467   | 1.0172    |
| 42    | 1.0000        | 0.6722      | 0.6652       | 0.0026     | 1.5349   | 1.6387    |
| 1234  | 0.9635        | 0.5796      | 0.7529       | 0.1450     | 0.6120   | 0.4161    |

---

### Statistiche aggregate (Test)

| Metrica        | Valore          |
|----------------|-----------------|
| Media          | 0.6196          |
| Dev. standard  | 0.0880          |
| Min–Max        | 0.4865 – 0.7529 |

---

### Osservazioni chiave

- **Varianza ampia**: il range di ≈ 0.27 punti di c-index (0.487 → 0.753) mostra che **il solo seed può ribaltare il ranking** fra modelli.
  
- **Seed 1234** è il migliore:
  - `c-index = 0.7529` supera:
    - ABMIL baseline (0.6851)
    - MCAT-MultimodalTopK (0.6618)
    - MCAT-Lite (0.5358)

- **Seed 42** (baseline usato altrove):  
  - c-index test = 0.665 (< 0.6851)  
  - Ma `log-rank p ≈ 0.0044` → **separazione clinicamente forte** nonostante la loss test più alta.

- **Pattern over-/under-fit**:
  - Seed 0 e 7 → validation ottima (0.80 / 0.77) ma calo sul test.
  - Seed 1 → crollo netto su val/test (0.65 → 0.49).
  - Seed 42 → memoriza perfettamente (`train = 1.0`) ma generalizza solo parzialmente.

- **Loss ≠ c-index**:
  - Solo seed **1234** combina **bassa loss (0.416)** e **alto c-index**.
  - Seed **42** ottiene buon ranking con la **peggior loss test (1.64)**.

---

### Confronto rapido con altri modelli

| Modello / Split                                | c-index Test |
|------------------------------------------------|--------------|
| ABMIL (seed 1234)                              | 0.7529       |
| ABMIL (baseline – α = 0, bin = 4, seed non fisso) | 0.6851    |
| MCAT-MultimodalTopK                            | 0.6618       |
| ABMIL (media 5 seed)                           | 0.6196       |
| MCAT-Lite                                      | 0.5358       |
| Genomics-only                                  | 0.6780       |

---

### Conclusioni

- L’inizializzazione **incide fortemente**: riportare i risultati di un solo seed può essere **fuorviante**.
- Con un seed favorevole (**1234**), ABMIL **supera tutti i modelli testati**.
- Tuttavia, la **media (0.620)** è **inferiore a MCAT-MultimodalTopK (0.662)** e al benchmark **ABMIL baseline (0.685)**.



