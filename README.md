# Bank Marketing Dataset — Machine Learning Project
## Overview
This project uses the Bank Marketing Dataset from the UCI Machine Learning Repository, which contains data from direct phone‑based marketing campaigns conducted by a Portuguese bank.
The goal is to predict whether a client will subscribe to a term deposit (y).
A full description of the dataset is available at source:
https://www.kaggle.com/datasets/sahistapatel96/bankadditionalfullcsv.

Through exploratory data analysis, several important characteristics of the dataset were identified:
- Strong seasonality across months, indicating that customer behavior changes significantly over time.
- Severe class imbalance, with the positive class being roughly four times smaller than the negative class.
- High correlation between socio‑economic indicators and the target, showing that macro‑economic conditions strongly influence subscription behavior.
- Many categorical features, some with uneven distribution across client groups, requiring careful preprocessing.

To preserve temporal structure and avoid data leakage, a time‑based train–test split by month was used. Because of the temporal nature of the data, resampling techniques such as SMOTE or undersampling were intentionally avoided, as they could distort seasonality and macro‑economic patterns.

Two preprocessing pipelines were implemented:
- Classical models pipeline for Logistic Regression, KNN, Decision Tree, and Random Forest
- Boosting‑specific pipeline for XGBoost, which handles categorical features differently

Several models were trained and compared: Logistic Regression, KNN, Decision Tree, Random Forest, and XGBoost.

Hyperparameter tuning was performed using:
- Grid Search (Decision Tree)
- Random Search (XGBoost)
- Hyperplot optimization (Random Forest and XGBoost)

The F1‑score was used as the primary evaluation metric to balance precision and recall under class imbalance.

Finally, feature importance analysis, SHAP interpretability, and misclassification analysis were conducted for the two best‑performing models — Random Forest and XGBoost — to understand their behavior, strengths, and limitations.

## Project Workflow
1. Exploratory Data Analysis
2. Modeling pipeline:
- Preprocessing for classical models
- Preprocessing for boosting models
- Model development
- Hyperparameter optimization
- Model evaluation
- Feature importance analysis
- SHAP analysis
- Misclassification analysis
- Conclusions

## Model Performance Summary
| Model                   | F1 (Train) | F1 (Val) | Recall (Train) | Recall (Val) | Precision (Train) | Precision (Val) |
|-------------------------|------------|----------|-----------------|--------------|--------------------|------------------|
| Logistic Regression     | 0.33       | 0.57     | 0.68            | 0.79         | 0.22               | 0.44             |
| Decision Tree (tuned)   | 0.32       | 0.57     | 0.69            | 0.78         | 0.21               | 0.45             |
| Random Forest (tuned)   | 0.47       | 0.58     | 0.63            | 0.76         | 0.37               | 0.48             |
| XGBoost (Hyperplot)     | 0.51       | 0.57     | 0.52            | 0.77         | 0.49               | 0.45             |
| XGBoost (Random Search) | 0.47       | 0.58     | 0.48            | 0.76         | 0.47               | 0.46             |

## Conclusion
- The results show that Random Forest and XGBoost are the two strongest models for this task, achieving the highest validation F1‑scores and demonstrating complementary strengths. 
- Random Forest is stable, conservative, and robust to temporal drift, producing fewer false positives and therefore offering a more cost‑efficient option for marketing teams. 
- XGBoost, on the other hand, is more expressive and sensitive to nonlinear patterns, especially macro‑economic indicators, which allows it to capture slightly more true subscribers but at the cost of a higher false‑positive rate.

Misclassification analysis confirms that both models struggle in situations where macro‑economic signals conflict with behavioral or demographic features. This indicates that the dataset contains complex interactions that are not fully captured by individual models.

To further improve performance, several enhancements are possible:
- introducing additional interaction features,
- adding temporal trend features to better model economic drift,
- enriching behavioral history features,
- applying target encoding for high‑cardinality categorical variables,
- incorporating customer segmentation or campaign‑level aggregates.

Based on the overall analysis, the most effective strategy would be to use a hybrid solution that combines Random Forest and XGBoost.

An ensemble approach such as averaging predicted probabilities or using a weighted voting scheme would leverage the stability of Random Forest and the expressive power of XGBoost.
Together, these models provide a richer and more complete understanding of customer behavior, reduce individual model weaknesses, and offer a more reliable prediction system for real‑world marketing applications.




