import os
import pandas as pd
from env import get_db_url
import env


# fx for acquiring telco_data:
def get_telco_data():
    telco_csv = 'telco_churn.csv'
    url = get_db_url('telco_churn')
    if os.path.isfile(telco_csv):
        return pd.read_csv(telco_csv)
    else:
        SQL = '''
    select * from customers
    join contract_types using (contract_type_id)
    join internet_service_types using (internet_service_type_id)
    join payment_types using (payment_type_id)
    '''
        df_telco = pd.read_sql(SQL, url)
        df_telco.to_csv(telco_csv)
        return df_telco


# be sure to join contract_types, internet_service_types, payment_types tables with the customers table from above
# a general edit of previous acquire and following prepare needs to occur.


# fx for prep_telco
def prep_telco():
    telco = get_telco_data()
    telco = telco.drop(columns=['internet_service_type_id', 'contract_type_id', 'payment_type_id','Unnamed: 0'])

    telco['gender_encoded'] = telco.gender.map({'Female': 1, 'Male': 0})
    telco['partner_encoded'] = telco.partner.map({'Yes': 1, 'No': 0})
    telco['dependents_encoded'] = telco.dependents.map({'Yes': 1, 'No': 0})
    telco['phone_service_encoded'] = telco.phone_service.map({'Yes': 1, 'No': 0})
    telco['paperless_billing_encoded'] = telco.paperless_billing.map({'Yes': 1, 'No': 0})
    telco['churn_encoded'] = telco.churn.map({'Yes': 1, 'No': 0})
    
    dummy_df = pd.get_dummies(telco[['multiple_lines', \
                              'online_security', \
                              'online_backup', \
                              'device_protection', \
                              'tech_support', \
                              'streaming_tv', \
                              'streaming_movies', \
                              'contract_type', \
                              'internet_service_type', \
                              'payment_type'
                            ]],
                              drop_first=True)
    telco = pd.concat( [telco, dummy_df], axis=1 )
# below is an edit to otherwise functioning fx to drop original non-encoded features
    telco = telco.drop(columns=['gender','partner','dependents','phone_service','multiple_lines','online_security','online_backup','device_protection','tech_support','streaming_tv','streaming_movies','paperless_billing','internet_service_type','payment_type'])
    telco = telco.drop(columns=['churn','contract_type'])
    return telco

# Split
from sklearn.model_selection import train_test_split
# fx for train-test-split of telco_data
def my_train_test_split(df, target=None):
    if target:
        train, test = train_test_split(df, test_size=.2, random_state=123, stratify=df[target])
        train, validate = train_test_split(train, test_size=.25, random_state=123, stratify=train[target])
    else:
        train, test = train_test_split(df, test_size=.2, random_state=123, stratify=df[target])
        train, validate = train_test_split(train, test_size=.25, random_state=123, stratify=train[target])
    return train, validate, test


