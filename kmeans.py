#TOHacks21 kmeans

#import
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

#read income file
df = pd.read_csv("income.csv")



#Set up clusters
km = KMeans(n_clusters=3)

#Fit age and income to clusters
y_predicted = km.fit_predict(df[['Age','Income($)']])

#Add predicted cluster to table
df['cluster']=y_predicted

#print df
print(df)
