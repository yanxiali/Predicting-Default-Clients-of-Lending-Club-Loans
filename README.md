# Predicting-Default-Clients-of-Lending-Club-Loans

In this project, I worked with the public LendingClub data, with a size of ~1.8 GB, containing 1.6 millions of loans from 2007 to 2017, each of which has 150 associated features. My goal was to build a predictive model that can predict whether or not a loan will be fully paid or charged off, in order to minimize the risks of loan defaults for the company. I also would like to find the most important factors in making decisions about lending.

I analyzed useful features and investigated the correlations among the features and between the features and the target variable. I used K-S tests to check whether the features have notably different distributions for the “fully paid” and “charged off” loans. According to the Pearson correlations between the features and the target variable, the most important variables for predicting charge-offs include the loan interest rate, loan term, the FICO score, and debt-to-income ratio. From feature importance analysis using the random forest classifier, I found that the most important features are interest rate and debt-to-income ratio.

I trained a logistic regression, a random forest, and a KNN to predict loan defaults and make loan-granting decisions. The models were all fine-tuned with grid search. I evaluated and compared the models using a cross-validated AUROC score on the training set. Finally, I found that all the three models have similar performance according to their AUROC scores on the training data. I eventually selected logistic regression because it ran the fast and returned an AUROC score of 0.70 on the test data. 

**Results**:
Please check out the jupyter notebook of the project [here](https://github.com/yanxiali/Predicting-Default-Clients-of-Lending-Club-Loans/blob/master/LC_Loan_full.ipynb). If you experience loading problems (as it is a big file), please take a look of a markdown copy of the project [here](https://github.com/yanxiali/Predicting-Default-Clients-of-Lending-Club-Loans/blob/master/results/LC_Loan_full.md)






