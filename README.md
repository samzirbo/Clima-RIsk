### Title:

**Quantifying Weather Effects on Healthcare: Dual-Model Approach for Hospital Admission Forecasting and Risk Prediction**

### Authors:

Samuel G.V. Zirbo, Bernadett S. Hoszu, Laura S. Dios, Adriana M. Coroiu, Adina E. Croitoru

### Conference:

28th International Conference on Knowledge-Based and Intelligent Information & Engineering Systems (KES 2024)

### Abstract:

This study investigates the impact of weather conditions on healthcare by utilizing a dual-model machine learning approach. The research focuses on two main tasks: predicting hospital admissions and assessing health risk for conditions such as heart failure, cerebral infarction, and respiratory failure. By leveraging meteorological data from weather stations and hospitalization data from major Romanian cities, the study employs multiple machine learning algorithms to evaluate the influence of environmental and health factors on these outcomes.

### Key Findings:

- **Hospital Admissions Forecasting:** XGBoost model achieved a mean absolute percentage error (MAPE) of 11.2%.
- **Health Risk Prediction:** XGBoost showed high recall rates for heart failure (94.1%), cerebral infarction (86.5%), and respiratory failure (65.1%).

### Methodology:

- **Data Sources:**
    - Meteorological data from official weather stations in Romania.
    - Hospitalization data from five major Romanian cities.
- **Machine Learning Algorithms:**
    - Linear and Logistic Regression
    - K-Neighbors
    - Decision Tree
    - Random Forest
    - Extreme Gradient Boosting (XGBoost)
- **Experimental Configurations:**
    - All features
    - Excluding weather parameters
    - Excluding medical information

### Detailed Results:

- Weather parameters had minimal impact compared to geographical, temporal, and medical information.
- Significant performance improvements were observed when including medical information, with age and pre-existing conditions being key predictors.

### Conclusions:

The study concludes that weather parameters have a minor role in predicting hospital admissions and health risks compared to other features such as age and existing medical conditions. Future work suggests expanding the feature set to include more climatic factors and exploring deep learning approaches.

### Limitations:

- The dataset is limited to five major Romanian cities, which may not represent other geographical regions.
- Weather parameters did not show significant correlation with health outcomes, potentially limiting the generalizability of the findings.
- The study focuses on a limited set of weather variables and health conditions, which may not capture the full spectrum of factors affecting hospital admissions and health risks.
- There is a need for more diverse datasets to validate the models and conclusions across different populations and environments.

### Contact Information:

- **Corresponding Author:** Samuel G.V. Zirbo
- **Email:** sam.zirbo@gmail.com
- **Phone:** +40-751-934-147
