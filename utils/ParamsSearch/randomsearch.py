import argparse
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import randint
from sklearn.model_selection import RandomizedSearchCV

def calculate_metric(file_path):
    df = pd.read_csv(file_path)

    nsga_indices = ['1', '101', '112', '139', '143', '157', '164', 
    '165', '17', '176', '178', '185', '189', '192', '207', '217', '230', 
    '235', '240', '242', '246', '262', '302', '309', '311', '314', '324', 
    '332', '333', '341', '352', '355', '359', '361', '363', '37', '382', 
    '393', '400', '421', '423', '430', '431', '436', '441', '451', '460', 
    '48', '485', '494', '497', '501', '502', '503', '504', '505', '59', 
    '87', '91', 'A', 'B', 'C', 'E', 'G']
    
    nsga_instances = df[df.iloc[:, 0].astype(str).isin(nsga_indices)] 
    moead_instances = df[~df.iloc[:, 0].astype(str).isin(nsga_indices)] 

    moead_indices = ['224', '151', '401', '413', '33', '234', '110', '492', '498', '175', '486', '259', 'D', '266', '456', 
    '12', '219', '10', '109', '353', '471', '28', '74', '213', '40', '495', '452', '32', '323', '312', '98', 
    '482', '399', '134', '330', '337', '376', '308', '43', '409', '261', '319', '412', '303', '487', '70', 
    '200', '244', '457', '170', '293', '467', '272', '105', '208', '179', '470', '114', '158', '57', '247', 
    '466', '472', '153', '288', '202', '198', '496', 'J', '68', '191', '390', '69', '236', '156', '46', '271', 
    '9', '162', '243', '469', '93', '256', 'F', '386', '276', '26', '295', '342', '348', '147', '15', '173', 
    '124', '142', '222', '260', '103', '35', '334', '306', '375', '107', '354', '104', '440', '372', '232', 
    '106', '255', '478', '120', '167', '435', '119', '65', '63', '113', '410', '350', '397', '346', '257', 
    '174', '161', '152', '392', '286', '181', '473', '378', '432', '77', '83', '437', '320', '381', '290', 
    '34', '64', '365', '403', '206', '67', '160', '414', '491', '51', '367', '190', '387', '195', '78', '483', 
    '38', '194', '117', '248', '252', '325', '304', '61', '340', '163', '449', '488', '241', '201', '458', 
    '339', '475', '116', '249', '299', '94', '128', '214', '80', '203', '434', '385', '277', '297', '216', '27', 
    '479', '220', '402', '212', '368', '357', '237', '29', '417', '329', '226', '53', '445', '425', '233', 
    '130', '81', '264', '131', '141', '199', '186', '159', '269', '298', '274', '88', '347', '96', '49', '404', 
    '209', '450', '318', '144', '47', '405', '45', '172', '238', '84', '132', '484', '127', '281', '433', '411', 
    '149', '477', '55', '317']

    moead_instances = moead_instances[moead_instances.iloc[:, 0].astype(str).isin(moead_indices)]

    scaler = StandardScaler()
    
    n_folds = 1
    scores = []
    
    for _ in range(n_folds):
        selected_nsga_indices_test = np.random.choice(nsga_indices, size=50, replace=False)
        selected_nsga_indices_train = list(set(nsga_indices) - set(selected_nsga_indices_test))

        #---------- BUILDING THE 90/10 TRAIN/TEST SPLIT ----------
        nsga_train = nsga_instances[nsga_instances.iloc[:, 0].astype(str).isin(selected_nsga_indices_train)]
        nsga_test = nsga_instances[nsga_instances.iloc[:, 0].astype(str).isin(selected_nsga_indices_test)]
        
        moead_train, moead_test = train_test_split(moead_instances, test_size=0.1, random_state=np.random.randint(0, 1000))
        
        train_data = pd.concat([nsga_train, moead_train])
        test_data = pd.concat([nsga_test, moead_test])
        
        X_train, y_train = train_data.iloc[:, 1:-2], train_data.iloc[:, -2:]
        X_test, y_test = test_data.iloc[:, 1:-2], test_data.iloc[:, -2:]
        
        #---------- NORMALIZATION OF FEATURES ----------
        scaler.fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        #---------- MULTIOUTPUT RANDOM FOREST REGRESSOR ----------   
        param_dist = {
            "estimator__n_estimators": randint(100, 1000),
            "estimator__max_depth": [None, 2, 4, 10, 20, 30],
            "estimator__min_samples_split": randint(2, 10),
            "estimator__min_samples_leaf": randint(1, 5)
        }
        
        base_model = MultiOutputRegressor(RandomForestRegressor(random_state=0))
        random_search = RandomizedSearchCV(base_model, param_distributions=param_dist, n_iter=10, cv=5, scoring='neg_mean_absolute_error', verbose=1, n_jobs=-1, random_state=0)
        random_search.fit(X_train_scaled, y_train)
        
        best_model = random_search.best_estimator_
        y_pred = pd.DataFrame(best_model.predict(X_test_scaled), columns=y_train.columns, index=y_test.index)

        print(random_search.best_params_)
        
        #---------- METRIC CALCULATION ----------        
        AS = y_train.columns[np.argmin(y_pred.values, axis=1)]
        VBS = y_test.idxmin(axis=1)
        SBS = y_train.mean().idxmin()
        
        rhv_AS = np.array([y_test.loc[y_test.index[i], AS[i]] for i in range(len(X_test))])
        rhv_SBS = np.array([y_test.loc[y_test.index[i], SBS] for i in range(len(X_test))])
        rhv_VBS = np.array([y_test.at[y_test.index[i], VBS.iloc[i]] for i in range(len(X_test))])
        
        rhv_AS_mean = rhv_AS.mean()
        rhv_SBS_mean = rhv_SBS.mean()
        rhv_VBS_mean = rhv_VBS.mean()
        
        if rhv_SBS_mean == rhv_VBS_mean:
            raise Exception("rhv_SBS_mean == rhv_VBS_mean")
        else:
            m = (rhv_AS_mean - rhv_VBS_mean) / (rhv_SBS_mean - rhv_VBS_mean)
        
        scores.append(m)
    
    print(np.mean(scores))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path", type=str)
    args = parser.parse_args()
    
    calculate_metric(args.file_path)