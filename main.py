import pandas as pd
import numpy as np
from matplotlib import pyplot as plt, pyplot
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, precision_recall_curve, auc, \
    classification_report, precision_recall_fscore_support, f1_score

dataset = pd.read_csv('dataset.csv')
dataset_test = pd.read_csv('dataset_test.csv')
#print(dataset.head())
#print(dataset.dtypes)

#Select features and target columns from dataset
features=dataset.iloc[:,1:-1]
target=dataset.iloc[:,-1]

#Select features and target columns from dataset_test
test_set=dataset_test.iloc[:,1:-1]
test_target=dataset_test.iloc[:,-1]

# print(final_features.shape)
# print(test_final.shape)
# print(target.shape)
# print(test_target.shape)


#Separate Categorical data and nunmerical data
features_num=features.select_dtypes(exclude=["object"])
features_cat=features.select_dtypes(include=["object"])

test_set_num=test_set.select_dtypes(exclude=["object"])
test_set_cat=test_set.select_dtypes(include=["object"])


#Convert categorical data into numerical columns. For eg- Check[Yes,no] will be Check_Yes[1,0] and Check_No[0,1]
features_cat_onehotencoded=pd.get_dummies(features_cat)
#print(features_cat_onehotencoded)
test_set_cat_onehotencoded=pd.get_dummies(test_set_cat)

#Concat the categorical and numerical data
final_features=pd.concat([features_num, features_cat_onehotencoded], axis=1)

test_final=pd.concat([test_set_num, test_set_cat_onehotencoded], axis=1)

#Calculate average transactional amount for calculation loss impact
txn_amount=test_final["Avg_transaction_amount_for_bill_payments_(INR)"]+test_final["Avg_amount_of_Retail_e-commerce_payments_(INR)"]
avg_txn_amount=txn_amount.mean()
#print("avg_txn_amount: "+str(avg_txn_amount))

#Apply Logistic Regression model
clf_logistic = LogisticRegression(solver='saga').fit(final_features, np.ravel(target))

#Prredict probability of default
preds=clf_logistic.predict_proba(test_final)
#print(preds)

#Plot Precision recall curve
# keep probabilities for the positive outcome only
lr_probs = preds[:, 1]
# predict class values
yhat=clf_logistic.predict(test_final)
lr_precision, lr_recall, _ = precision_recall_curve(test_target, lr_probs)
lr_f1, lr_auc = f1_score(test_target, yhat), auc(lr_recall, lr_precision)
# plot the precision-recall curves
no_skill = len(test_target[test_target==1]) / len(test_target)
pyplot.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
pyplot.plot(lr_recall, lr_precision, marker='.', label='Logistic')
# axis labels
pyplot.xlabel('Recall')
pyplot.ylabel('Precision')
# show the legend
pyplot.legend()
# show the plot
pyplot.show()

#Accuracy against test data
score=clf_logistic.score(test_final,test_target)
print("**** Accuracy against test data: "+ str(score))

# Create a dataframe for the probabilities of default
preds_df = pd.DataFrame(preds[:,1], columns = ['prob_default'])

# Reassign Risk based on the threshold
preds_df['Risk_1_or_0'] = preds_df['prob_default'].apply(lambda x: 1 if x > 0.4 else 0)

# Print the row counts for each Risk
# print("preds_df: ")
# print(preds_df.columns)
#print(preds_df['Risk_1_or_0'].value_counts())

# Print the classification report
target_names = ['Non-Default', 'Default']
print("**** Classification report ****")
print(classification_report(test_target, preds_df['Risk_1_or_0'], target_names=target_names))

# Plot ROC curve
prob_default = preds[:, 1]
fallout, sensitivity, thresholds = roc_curve(test_target, prob_default)
plt.plot(fallout, sensitivity, color = 'darkorange')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.title("ROC curve of the probabilities of default")
plt.xlabel("false positive rate (fall-out)")
plt.ylabel("true positive rate (sensitivity)")
plt.show()

#Accuracy score
auc1 = roc_auc_score(test_target, prob_default)
print("**** Accuracy score: "+str(auc1))


# Set the threshold for defaults to 0.4
preds_df['Risk_1_or_0'] = preds_df['prob_default'].apply(lambda x: 1 if x > 0.4 else 0)

# Print the confusion matrix
#   {True Neg(Predict=0, Actual=0) | False Pos(Predict=1, Actual=0)}
#  {False Neg(Predict=0, Actual=1) | True Pos(Predict=1, Actual=1)}

print("**** Confusion Matrix ****")
print(confusion_matrix(test_target,preds_df['Risk_1_or_0']))


#Estimated impact

preds_df['Risk_1_or_0'] = preds_df['prob_default'].apply(lambda x: 1 if x > 0.4 else 0)

# Store the number of loan defaults from the prediction data
num_defaults = preds_df['Risk_1_or_0'].value_counts()[1]

# Store the default recall from the classification report
default_recall = precision_recall_fscore_support(test_target,preds_df['Risk_1_or_0'])[1][1]

# Calculate the estimated impact of the new default recall rate
print("****Impact amount****")
print(num_defaults * avg_txn_amount * (1 - default_recall))
