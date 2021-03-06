# --------------
# Code starts here

avg_loan_amount =pd.pivot_table(banks, index = ['Gender','Married', 'Self_Employed'],
values = 'LoanAmount',
aggfunc = np.mean)

print(avg_loan_amount)


# code ends here



# --------------
# code starts here
banks = bank.drop(columns='Loan_ID')

print(banks.isnull().sum())
bank_mode = banks.mode()


banks.fillna(banks.mode().iloc[0],inplace=True)

print(banks.isnull().sum())

#code ends here


# --------------
# Import packages
import numpy as np
import pandas as pd
from scipy.stats import mode 
 



# code starts here
bank = pd.read_csv(path)
categorical_var=bank.select_dtypes('object')
numerical_var = bank.select_dtypes('number')
print(categorical_var)
print(numerical_var)


# --------------
# code starts here

loan_approved_se = banks[(banks['Self_Employed'] == 'Yes') & (banks['Loan_Status']=='Y')]
#print(loan_approved_se)

loan_approved_nse = len(banks[(banks['Self_Employed'] == 'No') & (banks['Loan_Status']=='Y')])
#print(loan_approved_nse)

Loan_Status = 614
percentage_se = (len(banks[(banks['Self_Employed'] == 'Yes') & (banks['Loan_Status']=='Y')]) / Loan_Status) *100
print(percentage_se)
percentage_nse = (len(banks[(banks['Self_Employed'] == 'No') & (banks['Loan_Status']=='Y')]) / Loan_Status) * 100
print(percentage_nse)
# code ends here


# --------------
# code starts here


# loan amount term 

loan_term = banks['Loan_Amount_Term'].apply(lambda x: int(x)/12 )


big_loan_term=len(loan_term[loan_term>=25])

print(big_loan_term)

# code ends here


# --------------
# code ends here

columns_to_show = ['ApplicantIncome', 'Credit_History']
 
loan_groupby=banks.groupby(['Loan_Status'])[columns_to_show]
print(loan_groupby)
# Check the mean value 
mean_values=loan_groupby.agg([np.mean])

print(mean_values)

# code ends here


