import pandas as pd
train = pd.read_csv('./OLID_dataset/olid_train_v2.csv',sep='\t')
val = train.sample(frac=0.065,replace=False)
final = pd.concat([train, val, val]).drop_duplicates(keep=False)

#sanity checks
assert len(final) == 13240 - len(val)
s1 = pd.merge(val, final, how='inner', on=['id'])
assert len(s1)==0

s1 = pd.merge(final, train, how='inner', on=['id'])
assert len(s1) == len(final)

s1 = pd.merge(val, train, how='inner', on=['id'])
assert len(s1) == len(val)

val.to_csv('./OLID_dataset/olid_val_v2.csv',sep='\t')
final.to_csv('./OLID_dataset/olid_train_v3.csv',sep='\t', index=False)

