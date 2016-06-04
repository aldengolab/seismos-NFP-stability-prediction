import pandas as pd
import sys

df = pd.read_csv(sys.argv[1])

original_length = len(df)
dfnew = df[df['2013_YOY_revenue_fell'].notnull()]
dfnew = dfnew[dfnew['2014_YOY_revenue_fell'].notnull()]
print("dropped {} percent  due to one of the two labels  missing".format((original_length -  len(dfnew)) / original_length))



total = len(dfnew)
TP = len(dfnew[dfnew['2013_YOY_revenue_fell'] & dfnew['2014_YOY_revenue_fell']])
FP = len(dfnew[dfnew['2013_YOY_revenue_fell'] & (dfnew['2014_YOY_revenue_fell'] == False)])
FN =  len(dfnew[(dfnew['2013_YOY_revenue_fell'] == False) & dfnew['2014_YOY_revenue_fell']])
TN = len(dfnew[(dfnew['2013_YOY_revenue_fell'] == False) & (dfnew['2014_YOY_revenue_fell'] == False)])

precision = TP / (TP + FP)
recall = TP /  (TP + FN)
accuracy = (TP + TN) / total
print("Total  was",  total)
print("True Positive Rate:", TP / total)
print("False Positive Rate:", FP / total)
print("True Negative Rate:", TN / total)
print("False Negative Rate:", FN / total)

print("Precision is:",  precision)
print("Recacll is:",  recall)
print( "Accuracy is", accuracy)


print("even more naive:")

TP2 = 0
FP2 = 0
FN2 = len(dfnew[dfnew['2014_YOY_revenue_fell']])
TN2 = len(dfnew[dfnew['2014_YOY_revenue_fell'] == False])

print("precision: 0")
print("recall: 0")
print("accuracy:", TN2 / total)



if __name__ == "__main__":
    if len(sys.argv) == 1:
        assert '.csv' in sys.argv[1]
        filename = int(sys.argv[1])
    else:
        print("This program analyzes a naive model on the dataset")
        print("Usage: python confusion_matrix.py <DATA FILE>")
