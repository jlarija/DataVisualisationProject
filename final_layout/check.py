import pickle


with open('df.pickle', 'rb') as dffile:
    df,variables_each_country = pickle.load(dffile)

print(df.columns)