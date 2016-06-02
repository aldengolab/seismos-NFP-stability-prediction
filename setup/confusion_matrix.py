import pandas as pd

df = pd.read_csv("../data/features_v1_5-25-16.csv")

dfnew = df[df['2013_rev_change'].notnull() & df['2014_rev_change'].notnull()]

drop2013 = dfnew['2013_rev_change'] <= -.2
drop2014 = dfnew['2014_rev_change'] <= -.2
notdrop2013 = dfnew['2013_rev_change'] > -.2
notdrop2014= dfnew['2014_rev_change'] > -.2

total = len(dfnew)
TP = len(dfnew[drop2013 & drop2014])
FP = len(dfnew[drop2013 & notdrop2014])
FN = len(dfnew[notdrop2013 & drop2014])
TN = len(dfnew[notdrop2013 & notdrop2014])

precision = TP / (TP + FP)
recall = TP /  (TP + FN)
print("Total  was",  total)
print("True Positive Rate:", TP / total)
print("False Positive Rate:", FP / total)
print("True Negative Rate:", TN / total)
print("False Negative Rate:", FN / total)

print("Precision is:",  precision)
print("Recacll is:",  recall)


print("even more naive:")

TP2 = 0
FP2 = 0
FN2 = len(dfnew[drop2014])
TN2 = len(dfnew[notdrop2014])

print("precision: 0")
print("recall: 0")
print("accuracy:", TN2 / total)


# print("Total Number of Drops Predicted for 2014 based on 2013"):
# print(yhat_drop_2014)
# print
