# Classification Project on Telco Customers
Demo on industrial application of classification ML models.

## Description:
Telco is a telecommunications company offering various products. This project uses information about this companies customers, such as various demographic data and information about the services provided to them. 

## Goals:
The goal of this project is to determine the drivers of churn, or how whether a customer stays or goes, from Telco customer data. After exploring the company data to find drivers of churn, a ML model to make reccomendations to the shareholders of Telco will created for generating actionable intelligence on customer churn.

## Beginning Assumptions:
Going into the project with churn and its contributing factors as the primary target, I will assess demographic data as well as details of the customers contracts.

## Methodology:
* Data acquired from MySQL 'telco_churn' database.
* Dataframe was 7043 rows, 25 columns before cleaning (row:customers, columns: variables).
* SQL query customers data, joining relevant tables on shared IDs, creating function to convert to dataframe.
* Clean data by dropping irrelevant columns.
* Encode useful categorical variables, removing original non-encoded.
* Split data (train/val/test) at &asymp; (60/20/20).
* Split stratification left ambiguous per use. 
* Model potential relationships with the use of visuals and graphs.
* Use various statistical tests and models to analyze combinations of variables against the target churn.
 - Hypotheses to test:
 * There is no significance between contract type and churn
 * There is no significance between single households (M&F) with no dependents and churn.
 * Tenure is unrelated to churn.
 * Monthly bill price is unrelated to churn.
 * Product subscriptions are unrelated to churn.
* After iterating models and selecting those with the most optimal evaluation metrics, evaluate with the best on test data.
* Synthesize exploration into actionable intelligence for shareholders.

## Data Dictionary:
| Variable Name                         | Description                                                      |
|---------------------------------------|------------------------------------------------------------------|
| churn_encoded                         | Binary encoding of churn status (0: Not Churn, 1: Churn)         |
| monthly_charges                       | The monthly charges for the service  (float)                     |
| tenure                                | Number of months the customer has been with the company (int)    |
| gender_encoded                        | Binary encoding of customer gender (0: Female, 1: Male)          |
| dependents_encoded                    | Binary encoding of customer having dependents (0: No, 1: Yes)    |
| internet_service_type_Fiber optic     | Binary encoding if customer has fiber optic internet service     |
| ...                                   | Various other encoded values are present in the DataFrame        |


## Reproduction of Results:


