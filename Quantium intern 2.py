import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from scipy import stats
from scipy.stats import norm, skew  # for some statistics
import re

# pd.set_option('display.max_columns', None)  # to display all rows and columns while printing.
# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_colwidth', -1)

# params = {'legend.fontsize': '15',
#           'axes.labelsize': 'x-large',
#           'axes.titlesize': 'x-large',
#           'xtick.labelsize': 'x-large',
#           'ytick.labelsize': 'x-large',
#           #  'axes.prop_cycle': plt.cycler(color = plt.cm.Set2.colors),
#           #  'image.cmap': 'Set2',
#           'figure.figsize': (18, 7)}
# plt.rcParams.update(params)
# plt.style.use('dark_background')  # to change the default values of plt to our interest.

# "C:\Users\adedi_tpk1ys1\OneDrive\Documents\PYTHON\Datasets\QVI_filtered_.xlsx"
# merged = pd.read_excel("C:/Users/adedi_tpk1ys1/OneDrive/Documents/PYTHON/Datasets/QVI_target_.xlsx")
# merged.pop('sm')
# print(merged.head())

# merged.insert(4, 'year_month', merged['date'].dt.to_period('M'))  # .dt.to_period('M')
# print(merged.head())
# print(merged.info())

# cols_with_changed_dtype = {'prod_name': 'category', 'prod_qty': 'category',
#                            'lifestage': 'category', 'premium_customer': 'category',
#                            'brand_name': 'category'}
# merged = merged.astype(cols_with_changed_dtype)
# print(merged.info())

# check = merged.groupby('store_nbr')['year_month'].nunique()
# check = check[check < 12]
# print('Stores with less than 12 month transaction data:')
# print(check)
# stores_with_less_than_12_months = check.index.to_list()
# del check

# indices_to_drop = merged[merged['store_nbr'].isin(stores_with_less_than_12_months)].index
# print(f'Shape before dropping: {merged.shape}\n')
# merged = merged.drop(indices_to_drop)
# print(f'Shape after dropping: {merged.shape}\n')
# print(f'Number of samples that are dropped: {len(indices_to_drop)}')

# check = merged[merged['store_nbr'].isin([77, 86, 88])].groupby(['store_nbr', 'year_month'])[
#     'tot_sales'].sum()  # code to plot
# colors = []
# for store, month in check.index:
#     if store == 77:
#         colors.append('b')
#     elif store == 86:
#         colors.append('g')
#     else:
#         colors.append('r')
#
# check.plot(kind='bar', color=colors, figsize=(12, 5))
# plt.show()
# del check, colors
# print(merged.head())

# merged['yearly_sale'] = merged.groupby('store_nbr')['tot_sales'].transform('sum')
# merged['yearly_custs'] = merged.groupby('store_nbr')['lylty_card_nbr'].transform('nunique')
# merged['monthly_sale'] = merged.groupby(['store_nbr', 'year_month'])['tot_sales'].transform('sum')
# merged['monthly_custs'] = merged.groupby(['store_nbr', 'year_month'])['lylty_card_nbr'].transform('nunique')
# df['monthly_txn'] = df.groupby(['store_nbr', 'year_month'])['txn_id'].agg({'txn_id': 'nunique'})

# avg_trans = merged.groupby('store_nbr').apply(
#     lambda subdf: (subdf['txn_id'].nunique() / subdf['yearly_custs'].unique()))
# avg_trans = avg_trans.astype('float64')
# merged['avg_txn_per_cust'] = merged['store_nbr'].map(avg_trans)
# print(merged.head())

# "pre_df" is the dataset which contains samples only before the trial period.
# "trial_df" is the dataset which contains samples of trial period.

# pre_df = merged[merged['date'] < "2019-02-01"]
# trial_df = merged[(merged['date'] > "2019-01-31") & (merged['date'] < "2019-05-01")]
#
# min_date_in_trial_df, max_date_in_trial_df = min(trial_df['date']), max(trial_df['date'])
# min_date_in_pre_df, max_date_in_pre_df = min(pre_df['date']), max(pre_df['date'])
# print(f'the trial_df dataframe consists of samples between {min_date_in_trial_df}, {max_date_in_trial_df}')
# print(f'the pre_df dataframe consists of samples between {min_date_in_pre_df}, {max_date_in_pre_df}')

# corr_mat = pre_df.corr()
# mask = np.triu(np.ones_like(corr_mat, dtype=bool))
# plt.subplots(figsize = (25, 15))
# sns.heatmap(corr_mat, mask = mask, cmap = 'coolwarm', annot = True)
# plt.xticks(rotation = 30)
# plt.show()
# metrics_cols' are the features for correlation and ranking between trial stores and control stores.
# metrics_cols = ['store_nbr', 'year_month', 'yearly_sale',
#                 'yearly_custs', 'monthly_sale', 'monthly_custs', 'avg_txn_per_cust']


# metrics_data = pre_trial_data.loc[:, metrics_cols]

# def extract_metrics(merged):
#     subdf = merged.loc[:, metrics_cols].set_index(['store_nbr', 'year_month']).sort_values(
#         by=['store_nbr', 'year_month'])
#     subdf.drop_duplicates(inplace=True, keep='first')
#     return subdf
#
#
# metrics_merged = extract_metrics(pre_df)


# print(metrics_merged.head())


# metrics_df.xs('2018-09', level=1)
# metrics_df.index.get_level_values('year_month').nunique()
# Function to find correlation between trial stores and control stores one by one.

# def calc_corr(trial_store):
#     '''
#     input: It takes one trial store to compare other stores with.
#     output: New dataframe with correlation and mean correlation.
#     '''
#     a = []
#     metrics = metrics_merged[['monthly_sale', 'monthly_custs']]
#     for i in metrics.index:
#         a.append(metrics.loc[trial_store].corrwith(metrics.loc[i[0]]))
#     subdf = pd.DataFrame(a)
#     subdf.index = metrics.index
#     subdf = subdf.drop_duplicates()
#     subdf.index = [s[0] for s in subdf.index]
#     subdf.index.name = "store_nbr"
#     subdf = subdf.abs()
#     subdf['mean_corr'] = subdf.mean(axis=1)
#     subdf.sort_values(by='mean_corr', ascending=False, inplace=True)
#     return subdf


"""Correlation with trial store: 77"""

# corr_77 = calc_corr(77).drop(77)
# corr_77 = corr_77.drop(77)
# print(corr_77.head(5))

# corr_77[corr_77['mean_corr'].abs() > 0.7].plot(kind='bar', rot=0, figsize=(18, 8))
# plt.title('Correlation of trial store 77 with other stores')
# plt.xlabel('store Number')
# plt.ylabel('Correlation Co-efficient')
# plt.show()

# The store '233' with the highest score is selected as the control store for trial store '77'.

# Now let's quantify how related it is to the trial store by using plots and some stats. Since monthly sales and
# customers are only parameters we can monitor. Therefore we'll just see these two parameters.


# fig, ax = plt.subplots()
# sns.distplot(metrics_merged.loc[77]['monthly_sale'], color='r', ax=ax)
# sns.distplot(metrics_merged.loc[233]['monthly_sale'], color='g', ax=ax)
# plt.legend(labels=['77', '233'])
# plt.show()

# From the above plot we can see that there is difference in monthly sale in both the stores.

# metrics_merged.loc[77]['monthly_sale'].plot(kind='bar', color='g')
# metrics_merged.loc[233]['monthly_sale'].plot(kind='bar', color='r', alpha=0.5)
# plt.xticks(rotation=0)
# plt.xlabel('Month')
# plt.ylabel('Monthly Sale')
# plt.legend(labels=(77, 233))
# plt.show()

# # Even though the monthly sale values are different but we see a similar trend in the sales through out the period.
#
# fig,ax = plt.subplots()
# sns.distplot(metrics_merged.loc[77]['monthly_custs'], color='r', ax=ax)
# sns.distplot(metrics_merged.loc[233]['monthly_custs'], color='g', ax=ax)
# plt.legend(labels=['77', '233'])
# plt.show()

# metrics_merged.loc[77]['monthly_custs'].plot(kind='bar', color='g')
# metrics_merged.loc[233]['monthly_custs'].plot(kind='bar', color='r', alpha=0.5)
# plt.xticks(rotation=0)
# plt.xlabel('Month')
# plt.ylabel('Monthly Sale')
# plt.legend(labels=(77, 232))
# plt.show()
#
# Even the trend in the number of customers every month follows a similar trend between the stores.

# Let our null hypothesis be that both the trial store and our selected control store are similar. Now if we want to
# reject the null hypothesis then we must have p-value close to zero.

from scipy.stats import ks_2samp, ttest_ind, t

# print(metrics_merged.head(2))
#
# cols_under_consideration = ['monthly_sale', 'monthly_custs']
# a = []
# for x in metrics_merged[cols_under_consideration]:
#     a.append(ks_2samp(metrics_merged.loc[77][x], metrics_merged.loc[233][x]))
# a = pd.DataFrame(a, index=cols_under_consideration)
# print(a.head())

# From the dataframe above we can say that both are similar (pvalues are high close to 1).
# Hence we cannot reject our null hypothesis.
"""Assessment of Trial."""

# Now we'll compare the trial store with the control store in the trial period i.e. from February 2019 to April 2019.

# trial_metrics_merged = extract_metrics(trial_df)
# print(trial_metrics_merged.head())

# b = []
# for x in trial_metrics_merged[cols_under_consideration]:
#     b.append(ks_2samp(trial_metrics_merged.loc[77][x], trial_metrics_merged.loc[233][x]))
# b = pd.DataFrame(b, index=cols_under_consideration)
# print(b.head())

# Since both the pvalues are >5 we reject the null hypothesis. Since both the stores are similar in pre-trial but not
# in trial period hence we reject the null hypothesis.

# Comparing each T-Value with 95% percentage significance critical t-value of 6 degrees of freedom (7 months of
# sample - 1)

# print('critical t-value for 95% confidence level:')
# print(t.ppf(0.95, 6))
# We can see that t-value is greater than 95 percentile for february to april.

# Therefore we can say that there was increase in sale in trial store than the control store during the trial period.

# Let's plot the means for both the stores in trial period.

# sns.distplot(trial_metrics_merged.loc[77]['monthly_sale'])
# sns.distplot(trial_metrics_merged.loc[233]['monthly_sale'])
# plt.legend(labels=['77','233'])
# plt.show()
#
# sns.distplot(trial_metrics_merged.loc[77]['monthly_custs'])
# sns.distplot(trial_metrics_merged.loc[233]['monthly_custs'])
# plt.legend(labels=['77','233'])
# plt.show()
# We can see that the distribution of monthly sale and monthly customers of both the stores in the trial period is
# much different than the distribution of monthly sale and monthly customers in pre-trial period.

# The results show that the trial store 77 is significantly different to its control store in the trial period as the
# trial store performance lies outside the 5% to 95% confidence interval of the control store in two of the three
# trial months.

# We can also see that there is significant increase in sales of chips in trial stores in the trial period.


"""Correlation with trial store: 86"""

# corr_86 = calc_corr(86).drop(86)
# print(corr_86.head())

# corr_86[corr_86['mean_corr'].abs() > 0.7].plot(kind = 'bar', rot = 0, figsize = (18, 8))
# plt.title('Correlation of trial store 86 with other stores')
# plt.xlabel('store Number')
# plt.ylabel('Correlation Co-efficient')
# plt.show()

# The store '155' with the highest score is selected as the control store for trial store '86'.

# Now let's quantify how related it is to the trial store by using plots and some stats. Since monthly sales and
# customers are only parameters we can monitor. Therefore we'll just see these two parameters.

# fig, ax = plt.subplots()
# sns.distplot(metrics_merged.loc[86]['monthly_sale'], color='r', ax=ax)
# sns.distplot(metrics_merged.loc[155]['monthly_sale'], color='g', ax=ax)
# plt.legend(labels=['86', '155'])
# plt.show()

# From the above plot we can see that there is difference in monthly sale in both the stores. But on avarage both the
# stores are similar.

# metrics_merged.loc[86]['monthly_sale'].plot(kind='bar', color='g')
# metrics_merged.loc[155]['monthly_sale'].plot(kind='bar', color='r', alpha=0.5)
# plt.xticks(rotation=0)
# plt.xlabel('Month')
# plt.ylabel('Monthly Sale')
# plt.legend(labels=(86, 155))
# plt.show()

# Even though the monthly sale values are different,but we see a similar trend in the sales throughout the period.

# fig, ax = plt.subplots()
# sns.distplot(metrics_merged.loc[86]['monthly_custs'], color = 'r', ax = ax)
# sns.distplot(metrics_merged.loc[155]['monthly_custs'], color = 'g', ax = ax)
# plt.legend(labels = ['86', '155'])
# plt.show()

# We can see that monthly customers are similar in both the stores.

# metrics_merged.loc[86]['monthly_custs'].plot(kind = 'bar', color = 'g')
# metrics_merged.loc[155]['monthly_custs'].plot(kind = 'bar', color = 'r', alpha = 0.5)
# plt.xticks(rotation = 0)
# plt.xlabel('Month')
# plt.ylabel('Monthly Sale')
# plt.legend(labels = (86, 155))
# plt.show()

# Even the trend in the number of customers every month follows a similar trend between the stores.

# Let our null hypothesis be that both the trial store and our selected control store are similar. Now if we want to
# reject the null hypothesis then we must have pvalue close to zero.
#
# cols_under_consideration = ['monthly_sale', 'monthly_custs']
# a=[]
# for x in metrics_merged[cols_under_consideration]:
#     a.append(ks_2samp(metrics_merged.loc[86][x], metrics_merged.loc[155][x]))
# a=pd.DataFrame(a, index = cols_under_consideration)
# print(a.head())

# From the dataframe above we can say that both are similar (pvalues are high close to 1). Hence, we cannot reject our
# null hypothesis.

"""Assessment of Trial."""

# Now we'll compare the trial store with the control store in the trial period i.e. from February 2019 to April 2019.
# b = []
# for x in trial_metrics_df[cols_under_consideration]:
#     b.append(ks_2samp(trial_metrics_df.loc[86][x], trial_metrics_df.loc[155][x]))
# b = pd.DataFrame(b, index = cols_under_consideration)
# print(b.head())

# Since all the p-values are high (say more than 0.05), we reject the null hypothesis i.e. there means are
# significantly different.

# Comparing each T-Value with 95% percentage significance critical t-value of 6 degrees of freedom (7 months of
# sample - 1)

# print('critical t-value for 95% confidence level:')
# print(t.ppf(0.95, 6))

# sns.distplot(trial_metrics_df.loc[86]['monthly_sale'])
# sns.distplot(trial_metrics_df.loc[155]['monthly_sale'])
# plt.legend(labels=['86','155'])
#
# sns.distplot(trial_metrics_df.loc[86]['monthly_custs'])
# sns.distplot(trial_metrics_df.loc[155]['monthly_custs'])
# plt.legend(labels=['86','155'])


# We can see that the distribution of monthly sale and monthly customers of both the stores in the trial period is
# much different than the distribution of monthly sale and monthly customers in pre-trial period. The results show
# that the trial store 86 is significantly different to its control store in the trial period as the trial store
# performance lies outside the 5% to 95% confidence interval of the control store in two of the three trial months.
# We can also see that there is significant increase in sales of chips in trial stores in the trial period.

"""Correlation with trial store: 88"""

# corr_88 = calc_corr(88).drop(88)
# corr_88.head()
#
# corr_88[corr_88['mean_corr'].abs() > 0.55].plot(kind = 'bar', rot = 0, figsize = (18, 8))
# plt.title('Correlation of trial store 88 with other stores')
# plt.xlabel('store Number')
# plt.ylabel('Correlation Co-efficient')
# plt.show()
#
# The store '14' has the highest score but we'll consider store '237' as the control store since the monthly sales
# is much correlated with it. Therefore store '237' is selected as the control store for trial store '88'. # Now
# let's quantify how related it is to the trial store by using plots and some stats. Since monthly sales and
# customers are only parameters we can monitor. Therefore we'll just see these two parameters.
#
# fig, ax = plt.subplots()
# sns.distplot(metrics_df.loc[88]['monthly_sale'], color = 'r', ax = ax)
# sns.distplot(metrics_df.loc[237]['monthly_sale'], color = 'g', ax = ax)
# plt.legend(labels = ['88', '237'])
# plt.show()
#
# From the above plot we can see that there is difference in monthly sale in both the stores. But on avarage both
# the stores are similar.
#
# metrics_df.loc[88]['monthly_sale'].plot(kind = 'bar', color = 'g')
# metrics_df.loc[237]['monthly_sale'].plot(kind = 'bar', color = 'r', alpha = 0.5)
# plt.xticks(rotation = 0)
# plt.xlabel('Month')
# plt.ylabel('Monthly Sale')
# plt.legend(labels = (88, 237))
# plt.show()
#
# Even though the monthly sale values are different but we see a similar trend in the sales through out the period.
#
# fig, ax = plt.subplots()
# sns.distplot(metrics_df.loc[88]['monthly_custs'], color = 'r', ax = ax)
# sns.distplot(metrics_df.loc[237]['monthly_custs'], color = 'g', ax = ax)
# plt.legend(labels = ['88', '237'])
# plt.show()
#
# We can see that monthly customers are similar in both the stores.
#
# metrics_df.loc[88]['monthly_custs'].plot(kind = 'bar', color = 'g')
# metrics_df.loc[237]['monthly_custs'].plot(kind = 'bar', color = 'r', alpha = 0.5)
# plt.xticks(rotation = 0)
# plt.xlabel('Month')
# plt.ylabel('Monthly Sale')
# plt.legend(labels = (88, 237))
# plt.show()
#
# Even the trend in the number of customers every month follows a similar trend between the stores.
#
# Let our null hypothesis be that both the trial store and our selected control store are similar. Now if we want
# to reject the null hypothesis then we must have pvalue close to zero.
#
# cols_under_consideration = ['monthly_sale', 'monthly_custs']
# a=[]
# for x in metrics_df[cols_under_consideration]:
#     a.append(ks_2samp(metrics_df.loc[88][x], metrics_df.loc[237][x]))
# a=pd.DataFrame(a, index = cols_under_consideration)
# print(a.head())
#
# From the dataframe above we can say that both are similar (pvalues are high close to 1). Hence we cannot reject
# our null hypothesis.
#
# """Assessment of Trial."""
#
# Now we'll compare the trial store with the control store in the trial period i.e. from Febraury 2019 to April 2019.
#
# b = []
# for x in trial_metrics_df[cols_under_consideration]:
#     b.append(ks_2samp(trial_metrics_df.loc[88][x], trial_metrics_df.loc[237][x]))
# b = pd.DataFrame(b, index = cols_under_consideration)
# print(b.head())
#
# Since all of the p-values are high (say more than 0.05), we reject the null hypothesis i.e. there means are
# significantly different. # Comparing each T-Value with 95% percentage significance critical t-value of 6 degrees of
# freedom (7 months of sample - 1)
#
# print('critical t-value for 95% confidence level:')
# print(t.ppf(0.95, 6))
#
#
# We can see that t-value is greater than 95 percentile for febraury to april.
#
# The results show that the trial in store 88 is significantly different to its control store in the trial period
# as the trial store performance lies outside of the 5% to 95% confidence interval of the control store in two of the
# three trial months.
#
# Let's plot the means for both the stores in trial period.
#
# sns.distplot(trial_metrics_df.loc[88]['monthly_sale'])
# sns.distplot(trial_metrics_df.loc[237]['monthly_sale'])
# plt.legend(labels=['88','237'])
#
# sns.distplot(trial_metrics_df.loc[88]['monthly_custs'])
# sns.distplot(trial_metrics_df.loc[237]['monthly_custs'])
# plt.legend(labels=['88','237'])
#
# We can see that the distribution of monthly sale and monthly customers of both the stores in the trial period is
# much different than the distribution of monthly sale and monthly customers in pre-trial period. # The results show
# that the trial store 88 is significantly different to its control store in the trial period as the trial store
# performance lies outside the 5% to 95% confidence interval of the control store in two of the three trial months. #
# We can also see that there is significant increase in sales of chips in trial stores in the trial period.
#
# """Conclusion"""
#
# The results for trial stores 77 and 88 during the trial period show a significant difference in at least two of
# the three trial months but this is not the case for trial store 86. We can check with the client if the
# implementation of the trial was different in trial store 86 but overall, the trial shows a significant increase in
# sales.
