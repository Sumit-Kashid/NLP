# Prior Authorization Readiness Classifier

I wanted to see if a fine-tuned biomedical language model could tell whether a
prior authorization (PA) request is ready to submit, using only lab results,
diagnosis and the requested procedure. The project builds leakage-free text inputs from
de-identified MIMIC-IV records, compares a TF-IDF baseline with three biomedical
BERT models, and looks at what the models actually learned.

Course project for BANA 8090 (Natural Language Processing), University of
Cincinnati, Spring 2026.

## Data

5,000 de-identified lab event records from MIMIC-IV (24 columns, 100 patients,
68 lab types). The target is `PriorAuth_Status`: Ready (2,964) or Not Ready
(2,036). The data is not in this repo because MIMIC-IV requires a PhysioNet
credentialed data use agreement.

## Approach

**Leakage removal.** The raw `Clinical_Comments` field was built from template
sentences ("Requesting prior authorization for...", "Documentation
incomplete...") that track the label almost directly. Only the diagnosis and
procedure names are pulled out with regex, and a check confirms none of the
template phrases remain.

**Structured notes.** Each record becomes one short note:

```
Diagnosis Obstructive Sleep Apnea. Procedure CPAP device approval.
Lab item 50983 value 130.0 mEq/L. Reference range 135.0 to 147.0.
Severity mildly below range. Flag abnormal. Priority routine.
```

Severity says how far a value falls outside its reference range (more than 30%
of the range width is "critically"), and flag and priority are normalized to
lowercase.

**Models.** All three transformers use the same settings and splits, so any
difference comes from pre-training:

| Model | Pre-training corpus |
|---|---|
| TF-IDF + logistic regression | none (baseline) |
| Bio_ClinicalBERT | MIMIC-III clinical notes (~880M tokens) |
| PubMedBERT | PubMed abstracts (~3.1B tokens) |
| BiomedBERT | PubMed abstracts + full text (~29B tokens) |

Stratified 70/15/15 split (3,430 / 735 / 736), learning rate 2e-5, 3 epochs with
early stopping, batch size 32, class-weighted cross entropy, decision threshold
tuned on the validation PR curve.

## Results (test set, n = 736)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| TF-IDF + LogReg | 50.8% | 59.8% | 51.8% | 55.5% | 0.502 |
| Bio_ClinicalBERT | | | | 74.4% | 0.486 |
| PubMedBERT | | | | 74.3% | 0.495 |
| BiomedBERT | 58.7% | 59.1% | 98.6% | 73.9% | 0.499 |
| *Always predict Ready* | *59.2%* | *59.2%* | *100%* | *74.4%* | *0.500* |

BiomedBERT confusion matrix: TN = 2, FP = 298, FN = 6, TP = 430.

## What the results mean

Every model, including the baseline, lands at a ROC-AUC of about 0.50, which is
no better than chance at ranking Ready above Not Ready. The transformers' F1 of
about 74% looks better than the baseline's 55%, but it matches what you get by
labeling every request Ready. BiomedBERT did nearly that: it flagged 298 of the
300 Not Ready cases as Ready.

My best guess is that once the template text was removed, the fields that were
left (diagnosis, procedure, one lab value, flag, priority) just don't carry
what decided the label. The original comments only made the task look easy
because the answer was basically written into the text. So the main thing I
took away from this project is how much leakage can hide: without that cleanup
step I would have reported a model that looked great and learned nothing
useful.

## What I'd do next

- Check for signal first: label rates by diagnosis, procedure, severity and
  priority, and a gradient boosted model on the structured fields. If those
  also sit at chance, more model tuning won't help.
- Bring in fields that plausibly drive PA decisions (prior treatments, payer,
  documentation status) without copying label text.
- Use 5-fold cross validation and report confidence intervals.
- Calibrate probabilities and route near-threshold cases to human review
  before any real use.

## Run it

Open `pa_readiness_classifier.ipynb` in Google Colab with a T4 GPU, run the
install cell, restart the runtime, then run the rest. Put the dataset at
`data/ClinicalDataProject.xlsx` or upload it when prompted. Training all three
models takes about 35 to 50 minutes.

## Tools

Python, Hugging Face Transformers and Datasets, PyTorch, scikit-learn, pandas,
matplotlib
