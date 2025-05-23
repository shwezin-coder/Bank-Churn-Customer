# Bank Customer Churn Prediction & Targeted Retention Strategies
-----

## Introduction

This project addresses the critical business challenge of customer attrition within a banking institution. Customer churn significantly impacts profitability and customer lifetime value. The traditional reactive approach to churn management is inefficient; hence, there was a clear need for a proactive, predictive system.

## Project Goal

The primary objective was to leverage comprehensive data analysis, statistical validation, and machine learning to accurately predict customer churn, identify the underlying factors driving attrition, and develop actionable, data-driven strategies to improve customer retention and engagement.

## Technologies Used

  * **Programming Language:** Python
  * **Key Libraries:** Pandas, NumPy, Scikit-learn, SciPy, Seaborn, Matplotlib, Plotly
  * **Interactive Applications:** Streamlit
  * **Data Source:** Excel

## Approach & Key Contributions

My approach followed a comprehensive data science pipeline, from meticulous data acquisition and rigorous cleaning to advanced modeling, insight generation, and interactive deployment:

### 1\. Data Acquisition & Initial Assessment

  * Initiated by extracting raw customer data from multiple Excel sheets (e.g., `Bank_Churn_Data_D....xlsx`, `Bank_Churn_Messy....xlsx`).
  * Performed initial assessments to understand data schemas, variable types, and potential inconsistencies across sheets.

### 2\. Comprehensive Data Cleaning & Preprocessing

  * **Data Merging:** Merged `customer info` and `account info` sheets based on common identifiers, ensuring data integrity across datasets.
  * **Data Type Validation:** Systematically checked and matched data types across merged columns, rectifying discrepancies.
  * **Duplicate Handling:** Identified and removed duplicate rows and duplicate columns to ensure data uniqueness and reduce redundancy.
  * **Column Renaming & Standardization:** Renamed columns for clarity and standardized categorical columns (e.g., `Gender`, `Geography`) to consistent formats.
  * **Missing Value Handling:** Identified missing values using `df.isnull().sum()`. For numerical features, missing values were **imputed with the median** to preserve distribution shape. For categorical features, missing values were **imputed with the mode** or a designated 'Unknown' category.
  * **Outlier Detection & Treatment:** Employed the **IQR (Interquartile Range) method** and visualized distributions (e.g., box plots using Seaborn) to detect outliers in numerical features. Outliers were then **capped at the 99th percentile or floored at the 1st percentile** to mitigate their impact on model performance while preserving data.
  * **Feature Engineering:** Created new, insightful features (e.g., `AvgSavingsToIncome` ratio, `HasCreditCard` flag) to enrich the dataset and enhance predictive power.

### 3\. Exploratory Data Analysis (EDA) & Visualization

  * After cleaning, I performed in-depth EDA and explored summary statistics on the merged dataset.
  * Key observations from EDA included:
      * A **100% correlation between credit card ownership and active status**, indicating a strong link between these two variables.
      * Segment-specific patterns were identified for `age`, `income`, `geography`, and `gender`.
      * Visualization of savings across geographies showed that **customers in Germany are predominantly saving customers**, while **Spain and France hold significant untapped potential** for savings products. France alone constituted 50% of the customer base.
      * Correlation analysis between numerical columns revealed **strong correlation only between `CreditCard` and `IsActiveMember`**; other numerical variables showed weak or no strong correlations with each other.
  * Visualizations were created using **Matplotlib, Seaborn, and Plotly** to effectively communicate these patterns and insights.

### 4\. Statistical Analysis & Hypothesis Testing for Churn Drivers

  * Initial visual analysis of churn status (e.g., between credit score groups) revealed significant differences, prompting formal statistical validation.
  * Applied rigorous **hypothesis testing** to statistically confirm relationships between variables and churn:
      * **T-test:** Conducted between `CreditScore` for churners vs. non-churners to assess mean differences.
      * **Mann-Whitney U test:** Applied to `EstimatedSalary` for churners vs. non-churners, given its non-normal distribution.
      * **Chi-squared (Chi²) test:** Used to assess the relationship between categorical variables and churn status.
  * **Key Findings from Statistical Tests on Churn Drivers:**
      * `NumOfProducts`, `Age Group`, `Geography`, `Active Status`, `Saving Group`, and `Credit Score Group` were **highly related to churn or stay status**.
      * `NumOfProducts` and `Age Group` showed a particularly strong relationship with churn.
      * `Geography`, `Active Status`, and `Saving Group` also exhibited strong correlations.
      * `Credit Score Group` was correlated but less strongly than the variables above.
      * Importantly, `Tenure` and `EstimatedSalary` were found **not to be significantly related** to churn.
  * This rigorous statistical analysis provided a deep, data-backed understanding of "why" customers churned, informing feature selection for predictive models.

### 5\. Feature Engineering for Predictive Modeling

  * Before model training, significant feature engineering was performed:
      * All categorical variables were transformed using **One-Hot Encoding**.
      * Numerical features (`Balance`, `CreditScore`, `EstimatedSalary`) were converted into **group/bin-based categorical data** (e.g., `CreditScoreGroup`, `IncomeGroup`) and then one-hot encoded. This was crucial as initial EDA indicated **non-linear relationships** with churn, making group-based features more effective than raw numerical inputs.

### 6\. Predictive Modeling & Detailed Churn Insights

  * **Model Training:** Trained predictive models using **Stratified K-Fold Cross-Validation** to ensure robust evaluation across different churn proportions and **RandomizedSearchCV for hyperparameter tuning**, ensuring optimal model performance across the entire dataset.
  * **Model Selection:** **Logistic Regression** was chosen as the primary model due to its **superior interpretability** and its performance being comparable to (or not significantly different from) Random Forest for this dataset.
  * **Performance:** The Logistic Regression model achieved an **accuracy of 0.83** and an F1-score of 0.60. While it successfully predicted **62% of churners**, it also indicated that 38% of churners were still being missed, highlighting a clear area for potential future model improvements (e.g., exploring advanced techniques or new data sources).
  * **Exported Insights:** Cleaned data (`cleaned_data.csv`), feature coefficients (`feature_coefficient.csv`), and churn probabilities were exported to CSV for further analysis and easy access by business teams.

### 7\. Strategic Customer Segmentation & Actionable Recommendations

  * Based on predictive probabilities, we extracted **potential high-risk customers (above 70% churn probability)** for proactive intervention.
  * Developed actionable insights and segmented customers for targeted interventions:
      * **High-Value Churned (3.14%):** Identified for immediate recovery efforts.
      * **High-Engagement Segment (3.4%):** Profiled as active customers with credit score \>700, salary & balance \>€100k (mostly male, age 24–60). Ideal targets for new product launches and promotions.
      * **High-Potential Segment (2.43%):** Defined as inactive but retained customers with strong credit (\>700), \~€100k salary & savings. Recommended for targeted credit card offers due to the 100% link observed between credit cards and activity.
      * **High-Value Segment (4.8%):** Comprising both active and inactive customers with credit score \>700, salary \>€100k. Inactive sub-segment targeted with credit card plans, while active sub-segment could be offered strict loans or adjusted savings rates to enhance retention.
  * Formulated concrete, data-backed recommendations:
      * Boost engagement with targeted **credit card offerings**, specifically for customers without cards, as they are often inactive and more likely to churn.
      * Focus **marketing and product development** on the critical **33–44 age group**.
      * Run **savings campaigns** in Spain & France, and conduct a competitor/service review in Germany to address regional churn.
      * Implement cross-selling strategies tailored by **gender and region** (e.g., more to females, encourage males to add products).
      * Prioritize early intervention on customers with **\>70% churn risk** by all relevant teams (Ops & Service).

### 8\. Interactive Application Development & Deployment (Streamlit)

  * To ensure insights were accessible and actionable for business stakeholders, I designed the user interface through **collaborative sketching sessions on Zoom Whiteboard**. This process was crucial for defining user flows, key visualizations, and dashboard layout.
  * Subsequently, I developed and deployed an interactive **Streamlit application** (`dashboard.py`). This powerful tool allows stakeholders to:
      * Visualize overall churn metrics and segment performance dynamically.
      * Explore influencing factors through interactive charts.
      * Input customer data for real-time churn risk prediction.
      * Access detailed customer tables (e.g., "High-Value Table," "Potential Customers Table"), empowering Marketing, Sales, Key Account, Operations, and Service teams for precise, targeted actions.

### Ethical Considerations & Future Steps:

  * Maintained strict **PII compliance** and emphasized responsible data use at all stages of the project lifecycle.
  * Outlined clear next steps for the bank, including continuous monitoring of churn metrics, refinement of the predictive model (especially focusing on improving recall for missed churners), and tracking the effectiveness of segment-specific campaign effectiveness.

## Impact & Outcome

This project provided the banking institution with a robust analytical framework and a user-friendly, interactive tool to proactively identify and engage at-risk customers. By translating complex data into actionable, statistically validated insights, it enabled the bank to optimize marketing and product development efforts, leading to more informed decisions that significantly improve customer retention, engagement, and overall profitability.

-----

## Live Demo

Experience the interactive Streamlit application:
https://bankchurnanalysis.streamlit.app/

-----

## Installation & Usage

To run this project locally, follow these steps:

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/BANK_CHURN_ANALYSIS.git
    cd BANK_CHURN_ANALYSIS
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: `venv\Scripts\activate`
    ```

3.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Place your data:**

      * Ensure your Excel data files (e.g., `Bank_Churn_Data_D....xlsx`, `Bank_Churn_Messy....xlsx`, `Bank_Churn.csv`) are placed in the `dataset/` directory.

5.  **Run the Streamlit application:**

    ```bash
    streamlit run dashboard.py
    ```

The application will open in your web browser.

## Project Structure

```
BANK_CHURN_ANALYSIS/
├── dataset/
│   ├── Bank_Churn_Data_D....xlsx
│   ├── Bank_Churn_Messy....xlsx
│   ├── Bank_Churn.csv
│   ├── cleaned_data.csv
│   └── feature_coefficient.csv
├── resources/
│   ├── datacleaning_steps.txt
├── dashboard.py
├── bank_churn_analysis.ipynb

└── requirements.txt
```

## Contributing

Feel free to fork this repository, open issues, or submit pull requests.

## Contact
For any questions or feedback, please contact Hnin Shwe Zin Hlaing at hninshwezinhlaing05062001@gmail.com.

