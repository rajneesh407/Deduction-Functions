import pandas as pd
import numpy as np

data=pd.read_pickle(r"C:\Users\rajneesh.jha\Downloads\deduction\GeorgiaPac.pkl")
print(data.shape)
# Replacing \N with np.nan`
data = data.replace('\\N', np.nan)
# Replacing None with np.nan
data = data.fillna(value=np.nan)
# Replacing the blank values
data = data.replace(r'^\s*$', np.nan, regex=True)
df=data


def get_summary(column_list):
    # Creating the summary dataframe
    summary_df = pd.DataFrame(columns=['column_name', 'null_percentage', 'no_of_uniques', 'value_counts'])
    summary_df['column_name'] = column_list

    # Creating summary
    for col in column_list:
        # null percentage
        summary_df.loc[summary_df.column_name == col, 'null_percentage'] = str(
            np.round((df[col].isnull().sum() / df.shape[0]) * 100, 2)) + ' % '

        # number of uniques
        summary_df.loc[summary_df.column_name == col, 'no_of_uniques'] = df[col].nunique()

        # value_counts

        ## Creating the value_counts dataframe
        vc = pd.DataFrame(df[col].value_counts(dropna=False).head(5)).reset_index()
        vc.columns = [col, 'count']
        vc['ratio'] = np.round(vc['count'] / len(df), decimals=2) * 100

        ## saving the dataframe in string list
        l = []
        for k in zip(vc[col].tolist(), vc['count'].tolist(), vc['ratio'].tolist()):
            l.append(str(k[0]) + ' --> ' + str(k[1]) + ' --> ' + str(k[2]) + ' % ')

        ## merging the string list
        s = ''
        for val in l:
            s = s + str(val) + ' || '

        # [:-3] to remove last || and space
        summary_df.loc[summary_df.column_name == col, 'value_counts'] = s[:-3]

    return summary_df

summary = get_summary(data.columns.tolist())
print(summary)
