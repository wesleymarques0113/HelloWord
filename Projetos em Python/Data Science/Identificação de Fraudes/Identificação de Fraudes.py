
import pandas as pd
import numpy as np

def blight_model():
    from sklearn.preprocessing import LabelEncoder
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.metrics import roc_auc_score
        
    train = pd.read_csv('train.csv', engine='python')
    test = pd.read_csv('test.csv', engine='python') 
    addresses = pd.read_csv('addresses.csv', engine='python') 
    latlons = pd.read_csv('latlons.csv', engine='python')

    train = train[np.isfinite(train['compliance'])]
    train = train[train.country == 'USA']
    test = test[test.country == 'USA']

    train = pd.merge(train, pd.merge(addresses, latlons, on='address'), on='ticket_id')
    test = pd.merge(test, pd.merge(addresses, latlons, on='address'), on='ticket_id')

    train.drop(['agency_name', 'inspector_name', 'violator_name', 'non_us_str_code', 'violation_description','grafitti_status', 
                'state_fee', 'admin_fee', 'ticket_issued_date', 'hearing_date', 'payment_amount', 'balance_due', 'payment_date', 
                'payment_status','collection_status', 'compliance_detail', 'violation_zip_code', 'country', 'address',
                'violation_street_number','violation_street_name', 'mailing_address_str_number', 'mailing_address_str_name', 
                'city', 'state', 'zip_code', 'address'], axis=1, inplace=True)
    
    label_encoder = LabelEncoder()
    for col in train.columns[train.dtypes == "object"]:
        train[col] = label_encoder.fit_transform(train[col])
    
    train['lat'] = train['lat'].fillna(method='pad')
    train['lon'] = train['lon'].fillna(method='pad') 
    test['lat'] = test['lat'].fillna(method='pad') 
    test['lon'] = test['lon'].fillna(method='pad') 
    train_columns = list(train.columns.values)
    train_columns.remove('compliance')
    test = test[train_columns]    
    
    X_train, X_test, y_train, y_test = train_test_split(train.ix[:, train.columns != 'compliance'], train['compliance'])
    rf = RandomForestRegressor()
    grid_values = {'n_estimators': [200], 'max_depth': [50]}
    grid_rf_auc = GridSearchCV(rf, param_grid=grid_values, scoring='roc_auc', cv=2)
    grid_rf_auc.fit(X_train, y_train)
    print('Model best parameter (max. AUC): ', grid_rf_auc.best_params_)
    print('Model score (AUC): ', grid_rf_auc.best_score_)


    for col in test.columns[test.dtypes == "object"]:
        test[col] = label_encoder.fit_transform(test[col])

    ans = pd.DataFrame(grid_rf_auc.predict(test), test.ticket_id) 
    return ans 

blight_model()


# In[ ]:




