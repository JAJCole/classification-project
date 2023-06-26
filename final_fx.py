import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

from acquire_prep import prep_telco
from acquire_prep import my_train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import chi2_contingency
from scipy.stats import mannwhitneyu


import warnings
warnings.filterwarnings("ignore")

# Create variable for df manipulation
df = prep_telco()

def vis_1():
    # split
    train, validate, test = my_train_test_split(df,'churn_encoded')
    # Grouped data for contract_type_Owo year
    grouped_data_1 = train.groupby('contract_type_One year')['churn_encoded'].mean()
    # Grouped data for contract_type_Two year
    grouped_data_2 = train.groupby('contract_type_Two year')['churn_encoded'].mean()
    # Creating subplots
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    # Plotting the bar plot for contract_type_One year
    axs[0].bar(grouped_data_1.index, grouped_data_1)
    axs[0].set_xlabel('Contract Type One year')
    axs[0].set_ylabel('Churn Rate')
    #axs[0].set_title('Churn Rate by Contract Type')
    # Plotting the bar plot for contract_type_Two year
    axs[1].bar(grouped_data_2.index, grouped_data_2)
    axs[1].set_xlabel('Contract Type Two year')
    axs[1].set_ylabel('Churn Rate')
    #axs[1].set_title('Churn Rate by Contract Type')
    # Adjusting the layout
    plt.tight_layout()
    # title
    plt.title('Churn Rate by Contract type')
    # Displaying the plots
    return plt.show


def chi_1():
    # split
    train, validate, test = my_train_test_split(df,'churn_encoded')


    # create fx to undo binary to fit to chi squared
    train['contract_type'] = train[['contract_type_One year', 'contract_type_Two year']].apply(lambda x: 'contract_type_One year' if x['contract_type_One year'] == 1 else ('contract_type_Two year' if x['contract_type_Two year'] == 1 else 'month-to-month'), axis=1)
    # crosstab contract types and create contigency table with target churn
    contingency_table = pd.crosstab(train['contract_type'], train['churn_encoded'])
    # compute and display values of test
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
    print("Chi-squared statistic: {}\nP-value: {}\nDegrees of freedom: {}\nExpected frequencies:\n {}".format(chi2, p_value, dof, expected))
    return
####


def vis_2():
    # analyze only train data
    train, validate, test = my_train_test_split(df,'churn_encoded')
    df_b = train[['monthly_charges', 'churn_encoded']]
    # Set the figure size
    plt.figure(figsize=(8, 6))
    # Create the box plot
    sns.boxplot(x='churn_encoded', y='monthly_charges', data=df_b)
    # Calculate and plot the average monthly price
    average_monthly_price = df_b['monthly_charges'].mean()
    plt.axhline(y=average_monthly_price, color='r', label='Average Monthly Price')
    # Set the plot labels and title
    plt.xlabel('Churn')
    plt.ylabel('Monthly Charges')
    plt.title('Box Plot of Churn vs Monthly Charges')
    # Add a legend
    plt.legend()
    # Display the plot
    return plt.show()



# for assessing contract type and churn w chi2:
def chi_2():
     # create subset of df with columns of interest
    df_f = df[['monthly_charges', 'churn_encoded']]
    # split
    train, validate, test = my_train_test_split(df_f,'churn_encoded')
    # Create contingency table
    contingency_table = pd.crosstab(df_f['monthly_charges'], df_f['churn_encoded'])
    # Perform chi-squared test
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
    # Print the results
    print("Chi-squared statistic: {:.6f}".format(chi2))
    print("P-value: {:.6f}".format(p_value))
    print("Degrees of freedom: {}".format(dof))
    print("Expected frequencies:\n", expected)
    return




# below is a rf fx (might remove)
# for contract type-churn
def rf_1():
    # create subset of df with columns of interest
    df_f = df[['monthly_charges', 'churn_encoded']]
    # split
    train, validate, test = my_train_test_split(df_f,'churn_encoded')
    # drop targets and nulls from feature train/val/test, set target
    x_train = train.drop(columns=['churn_encoded']).dropna()
    y_train = train.churn_encoded.dropna()
    x_val = validate.drop(columns=['churn_encoded']).dropna()
    y_val = validate.churn_encoded
    x_test = test.drop(columns=['churn_encoded']).dropna()
    y_test = test.churn_encoded
    # create object
    rf = RandomForestClassifier(bootstrap=True, 
                                class_weight=None, 
                                criterion='gini',
                                min_samples_leaf=1,
                                n_estimators=100,
                                max_depth=20, 
                                random_state=123)
    rf.fit(x_train, y_train)
    # make predictions
    y_pred = rf.predict(x_train)
    # estimate probability of survive
    y_pred_proba = rf.predict_proba(x_train)
    print('Accuracy of random forest classifier on training set: {:.2f}'
     .format(rf.score(x_train, y_train)))
    print(classification_report(y_train, y_pred))
    return



# mann whitney test fx for:
# 'of those paying higher, do they have fiber optic?'
def m_w_1():
    df_fiber_optic = df[df['internet_service_type_Fiber optic'] == 1]
    df_non_fiber_optic = df[df['internet_service_type_Fiber optic'] == 0]

    # Check if any samples exist in both groups
    if len(df_fiber_optic) > 0 and len(df_non_fiber_optic) > 0:
        x = df_fiber_optic['monthly_charges']
        y = df_non_fiber_optic['monthly_charges']

        # Perform the Mann-Whitney U test
        stat, p_value = mannwhitneyu(x, y)
        # Print the results
        print("Mann-Whitney U test statistic:", stat)
        print("P-value:", p_value)
        print("Average monthly charge (Fiber optic):", np.mean(x))
        print("Average monthly charge (Non-Fiber optic):", np.mean(y))
        return
    else:
        print("No samples found in one or both groups.")
        return
    

# is tenure related to churn?
def vis_3():
    train, validate, test = my_train_test_split(df,'churn_encoded')
    df_c = train[['tenure', 'churn_encoded']]
    sns.boxplot(x='churn_encoded', y='tenure', data=df_c)
    plt.xlabel('Churn')
    plt.ylabel('Tenure')
    plt.title('Tenure of Customers by Churn')
    return plt.show()

def m_w_2():
    # Separate tenure values for churned and non-churned customers
    tenure_churned = df[df['churn_encoded'] == 1]['tenure']
    tenure_non_churned = df[df['churn_encoded'] == 0]['tenure']

    # Perform Mann-Whitney U test
    statistic, p_value = stats.mannwhitneyu(tenure_churned, tenure_non_churned)

    # Display the test statistic and p-value
    print("Mann-Whitney U test statistic:", statistic)
    print("P-value:", p_value)
    return


# models fx (train)
def model_fx():
    df = prep_telco()
    df = df[['contract_type_One year', 'contract_type_Two year', 'churn_encoded', 'tenure', 'internet_service_type_Fiber optic']]
    #my_train_test_split(df, target='churn_encoded')
    train, validate, test = my_train_test_split(df,'churn_encoded')
    # this split removes further objects from train,val, test for logistic regression error avoidance
    x_train = train.drop(columns=['churn_encoded']).dropna()
    y_train = train.churn_encoded.dropna()
    x_val = validate.drop(columns=['churn_encoded']).dropna()
    y_val = validate.churn_encoded
    x_test = test.drop(columns=['churn_encoded']).dropna()
    y_test = test.churn_encoded
    # log. reg model
    reg_o = LogisticRegression()
    reg_o.fit(x_train, y_train)
    y_pred = reg_o.predict(x_train)
    y_pred_proba = reg_o.predict_proba(x_train)
    print('Accuracy of Log. Reg. object on train set: {:.3f}'
     .format(reg_o.score(x_train, y_train)))
    print(classification_report(y_train, y_pred))
    # dt model
    tree = DecisionTreeClassifier(max_depth=3, random_state=123)
    tree = tree.fit(x_train, y_train)
    y_pred = tree.predict(x_train)
    print('Accuracy of DT object on train set: {:.3f}'.format(tree.score(x_train, y_train)))
    print(classification_report(y_train, y_pred))
    # rf model
    rf = RandomForestClassifier(bootstrap=True, 
                            class_weight=None, 
                            criterion='gini',
                            min_samples_leaf=1,
                            n_estimators=100,
                            max_depth=20, 
                            random_state=123)
    rf.fit(x_train, y_train)
    # make predictions
    y_pred = rf.predict(x_train)
    # estimate probability of survive
    y_pred_proba = rf.predict_proba(x_train)
    print('Accuracy of random forest classifier on training set: {:.2f}'
     .format(rf.score(x_train, y_train)))
    print(classification_report(y_train, y_pred))
    # KNN
    knn = KNeighborsClassifier(n_neighbors=10)
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_train)
    y_pred_proba = knn.predict_proba(x_train)
    print('Accuracy of KNN classifier on training set: {:.2f}'
     .format(knn.score(x_train, y_train)))
    print(classification_report(y_train, y_pred))
    return
#####
# models fx (val)
def model_fx_v():
    df = prep_telco()
    df = df[['contract_type_One year', 'contract_type_Two year', 'churn_encoded', 'tenure', 'internet_service_type_Fiber optic']]
    #my_train_test_split(df, target='churn_encoded')
    train, validate, test = my_train_test_split(df,'churn_encoded')
    # this split removes further objects from train,val, test for logistic regression error avoidance
    x_train = train.drop(columns=['churn_encoded']).dropna()
    y_train = train.churn_encoded.dropna()
    x_val = validate.drop(columns=['churn_encoded']).dropna()
    y_val = validate.churn_encoded
    x_test = test.drop(columns=['churn_encoded']).dropna()
    y_test = test.churn_encoded
    # log. reg model
    reg_o = LogisticRegression()
    reg_o.fit(x_val, y_val)
    y_pred = reg_o.predict(x_val)
    y_pred_proba = reg_o.predict_proba(x_val)
    print('Accuracy of Log. Reg. object on train set: {:.3f}'
     .format(reg_o.score(x_val, y_val)))
    print(classification_report(y_val, y_pred))
    # dt model
    tree = DecisionTreeClassifier(max_depth=3, random_state=123)
    tree = tree.fit(x_val, y_val)
    y_pred = tree.predict(x_val)
    print('Accuracy of DT object on train set: {:.3f}'.format(tree.score(x_train, y_train)))
    print(classification_report(y_val, y_pred))
    # rf model
    rf = RandomForestClassifier(bootstrap=True, 
                            class_weight=None, 
                            criterion='gini',
                            min_samples_leaf=1,
                            n_estimators=100,
                            max_depth=20, 
                            random_state=123)
    rf.fit(x_val, y_val)
    # make predictions
    y_pred = rf.predict(x_val)
    # estimate probability of survive
    y_pred_proba = rf.predict_proba(x_val)
    print('Accuracy of random forest classifier on training set: {:.2f}'
     .format(rf.score(x_val, y_val)))
    print(classification_report(y_val, y_pred))
    # KNN
    knn = KNeighborsClassifier(n_neighbors=10)
    knn.fit(x_val, y_val)
    y_pred = knn.predict(x_val)
    y_pred_proba = knn.predict_proba(x_val)
    print('Accuracy of KNN classifier on training set: {:.2f}'
     .format(knn.score(x_val, y_val)))
    print(classification_report(y_val, y_pred))
    return



# rf won highest accuracy, so test:
def model_fx_t():
    df = prep_telco()
    df_t = df[['contract_type_One year', 'contract_type_Two year', 'churn_encoded', 'tenure', 'internet_service_type_Fiber optic','customer_id']]
    #my_train_test_split(df, target='churn_encoded')
    train, validate, test = my_train_test_split(df_t,'churn_encoded')
    # this split removes further objects from train,val, test for logistic regression error avoidance
    x_test = test
    customer_ids = x_test['customer_id']  # Store customer IDs
    x_test = test.drop(columns=['churn_encoded','customer_id']).dropna()
    y_test = test.churn_encoded
    rf = RandomForestClassifier(bootstrap=True, 
                            class_weight=None, 
                            criterion='gini',
                            min_samples_leaf=1,
                            n_estimators=100,
                            max_depth=20, 
                            random_state=123)
    rf.fit(x_test, y_test)
    # make predictions
    y_pred = rf.predict(x_test)
    # estimate probability of survive
    y_pred_proba = rf.predict_proba(x_test)[:, 1]  # Select positive class probabilities (churn)
    # create df for predictions/id csv
    predictions_df = pd.DataFrame({'customer_id': customer_ids, 'probability_of_churn': y_pred_proba, 'prediction_of_churn': y_pred})
    predictions_df.to_csv('predictions.csv', index=False)
    # display report
    print('Accuracy of random forest classifier on training set: {:.2f}'
     .format(rf.score(x_test, y_test)))
    print(classification_report(y_test, y_pred))
    return

