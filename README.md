# Machine Failure Prediction

Predicting machine failures from nine sensor readings, including air quality (AQ) and volatile organic compounds
(VOC) measured next to each machine. The project combines exploratory analysis and hypothesis tests with
dimensionality reduction (PCA) and a tuned Random Forest.

## Results

Test set of 189 machines (20 % hold-out):

| Metric | Value |
|---|---:|
| Accuracy | 87.3 % |
| ROC AUC | 0.943 |
| Precision / recall (both classes) | ≈ 0.87 |
| Errors | 14 false alarms, 10 missed failures |

Cross-validated ROC AUC on the training set was 0.965.

## Key findings

- **VOC is the strongest single predictor** (correlation 0.80 with failure): failure rates rise from under 10 % at
  VOC levels 0–2 to over 80 % at levels 5–6. AQ shows the same pattern more weakly (0.58).
- Mann-Whitney U and chi-square tests confirm both relationships (p < 10⁻⁷⁰).
- The sensors are strongly correlated with each other (VIF above 10 for AQ, CS, IP and RP), so the nine readings are
  reduced to five principal components (77 % of the variance). The first component, driven by VOC, temperature, AQ
  and the ultrasonic sensor, carries most of the predictive information.

## Data

`data/data.csv`: 944 records with nine sensor readings (footfall, temperature mode, AQ, ultrasonic sensor, current,
VOC, rotational position, input pressure, temperature) and a failure flag (42 % failures).

## Notebooks

| Notebook | Content |
|---|---|
| [`EDA.ipynb`](EDA.ipynb) | Distributions, correlations, failure rates by VOC and AQ level, hypothesis tests, VIF, PCA |
| [`Modeling.ipynb`](Modeling.ipynb) | Model comparison with PyCaret, Random Forest tuned with randomized search, evaluation |

## Run it

```bash
pip install -r requirements.txt
jupyter notebook EDA.ipynb
```

## Limitations

With 944 records the test set is small, so the scores have wide uncertainty. PCA makes the model harder to interpret
than one built directly on the sensor readings.

## License

MIT
