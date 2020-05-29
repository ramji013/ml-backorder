import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
import pickle as pkl

back_order = pd.read_csv('Training_Dataset_v2.csv')

back_order_new = back_order[['national_inv', 'lead_time', 'in_transit_qty', 'forecast_3_month', 'forecast_6_month',
                             'forecast_9_month', 'sales_1_month', 'sales_3_month', 'sales_6_month', 'sales_9_month',
                             'min_bank', 'pieces_past_due', 'perf_6_month_avg', 'perf_12_month_avg']]

#rename columns to understand it better
back_order_new.rename(columns = {'national_inv' : 'current_inventory_level' , 'lead_time': 'transit_time', 'min_bank': 'min_stock'}, inplace= True)


#to know the list of columns has null value
null_columns=back_order_new.columns[back_order_new.isna().any()]
print(back_order_new[null_columns].isna().sum())

#it was identified that transit_time has more number of null values. now, going to fill the null values with mean
back_order_new['transit_time'].fillna(back_order_new['transit_time'].mean(), inplace=True)
print(back_order_new[null_columns].isna().sum())

# one row with null value found. remove the null row
back_order_new.dropna(inplace=True)


#to know how many values went back order
print(back_order['went_on_backorder'].value_counts())

y= back_order['went_on_backorder'].dropna()


X_train, X_test, y_train, y_test = train_test_split(back_order_new, y, test_size=0.3, random_state = 0)
clf= SGDClassifier()
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

print(accuracy_score(y_test, y_pred))

out = open('../model/classifier.pkl','wb')
pkl.dump(clf, out)
out.close()