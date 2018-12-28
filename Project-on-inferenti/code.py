# --------------
import pandas as pd
import scipy.stats as stats
import math
import numpy as np
import warnings

warnings.filterwarnings('ignore')
#Sample_Size
sample_size=2000

#Z_Critical Score
z_critical = stats.norm.ppf(q = 0.95)  


# path        [File location variable]

#Code starts here
data = pd.read_csv(path)
data_sample = data.sample(n=sample_size, random_state = 0)
sample_mean = data_sample['installment'].mean()

sample_std = data_sample['installment'].std()

margin_of_error = z_critical*(sample_std/np.sqrt(sample_size))
print(margin_of_error)
confidence_interval = (sample_mean - margin_of_error , sample_mean + margin_of_error)
print(confidence_interval)

true_mean = data['installment'].mean()
if true_mean >= confidence_interval[0] and true_mean <= confidence_interval[1]:
    print('valid')
else:
    print('invalid')


# --------------
import matplotlib.pyplot as plt
import numpy as np

#Different sample sizes to take
sample_size=np.array([20,50,100])
fig, axes = plt.subplots(nrows=3,ncols=1)
#Code starts here
for i in range(len(sample_size)):
    m =list()
    for j in range(1000):
        sample_data = data.sample(n=sample_size[i],random_state = 0)
        m.append(sample_data['installment'].mean())

    mean_series = pd.Series(m)
    plt.plot(mean_series[i])


# --------------
#Importing header files

from statsmodels.stats.weightstats import ztest
data['int.rate'] = data['int.rate'].map(lambda x: x.rstrip('%'))
data['int.rate'] = data['int.rate'].astype('float64')
data['int.rate'] = data['int.rate']/100
x1 = data[data['purpose']=='small_business']['int.rate']

mean_value = data['int.rate'].mean()
z_statistic,p_value = ztest(x1,value=mean_value, alternative='larger')
print(z_statistic)
print(p_value)

if p_value < 0.5:
    hypothesis = 'Reject'
else:
    hypothesis = 'Accept'

print(hypothesis)
#Code starts here



# --------------
#Importing header files
from statsmodels.stats.weightstats import ztest
x1=data[data['paid.back.loan']=='No']['installment']
x2=data[data['paid.back.loan']=='Yes']['installment']
z_statistic,p_value = ztest(x1,x2)
print(p_value)
#Code starts here
if p_value < 0.5:
    print('reject')
else:
    print('accept')



# --------------
#Importing header files
from scipy.stats import chi2_contingency

#Critical value 
critical_value = stats.chi2.ppf(q = 0.95, # Find the critical value for 95% confidence*
                      df = 6)   # Df = number of variable categories(in purpose) - 1

#Code starts here
yes = data[data['paid.back.loan']=='Yes']['purpose'].value_counts()
no = data[data['paid.back.loan']=='No']['purpose'].value_counts()

y=yes.transpose()
n=no.transpose()

observed = pd.concat([y,n], axis=1, keys = ['Yes','No'])
chi2,p,dof,ex = chi2_contingency(observed)

if chi2 > critical_value:
    print('Reject')
else:
    print('accept')


