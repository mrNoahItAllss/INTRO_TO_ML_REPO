#imports:
from sklearn import preprocessing 
import os
import pandas as pd

###########################################################################################################

def compute_cost(X, y, theta): 
    """ 
    Compute cost for linear regression. 
    
    Input Parameters 
    ---------------- 
    X : 2D array where each row represent the training example and each column represent 
        m= number of training examples 
        n= number of features (including X_0 column of ones) 
    y : 1D array of labels/target value for each traing example. dimension(1 x m) 
    
    theta : 1D array of fitting parameters or weights. Dimension (1 x n) 
    
    Output Parameters 
    ----------------- 
    J : Scalar value. 
    """ 
    m = len(y)
    predictions = X.dot(theta) 
    errors = np.subtract(predictions, y) 
    sqrErrors = np.square(errors) 
    J = 1 / (2 * m) * np.sum(sqrErrors) 
    
    return J 

########################################################################################################### 

def gradient_descent(X, y, theta, alpha, iterations): 
    """ 
    Compute cost for linear regression. 
    
    Input Parameters 
    ---------------- 
    X : 2D array where each row represent the training example and each column represent 
        m= number of training examples 
        n= number of features (including X_0 column of ones) 
    y : 1D array of labels/target value for each traing example. dimension(m x 1) 
    theta : 1D array of fitting parameters or weights. Dimension (1 x n) 
    alpha : Learning rate. Scalar value 
    iterations: No of iterations. Scalar value.  
    
    Output Parameters 
    ----------------- 
    theta : Final Value. 1D array of fitting parameters or weights. Dimension (1 x n) 
    cost_history: Conatins value of cost for each iteration. 1D array. Dimansion(m x 1)   
    """ 
    theta_history = np.zeros((iterations, len(theta)))
    X = pd.concat([pd.Series(1, index=X.index, name='x_0'), X], axis=1)
    m = len(y)
    cost_history = np.zeros(iterations) 
    for i in range(iterations): 
        predictions = X.dot(theta) 
        errors = np.subtract(predictions, y) 
        sum_delta = (alpha / m) * X.transpose().dot(errors); 
        theta = theta - sum_delta; 
        theta_history[i, :] = theta 
        cost_history[i] = compute_cost(X, y, theta)   
    
    return theta, cost_history, theta_history

###########################################################################################################

def cat_to_useful(df):
    newDf = df
    col = df.columns.tolist()
    types = df.dtypes
    for i in range(len(col)):

        if types[i] == 'object' or types[i] == 'str':
            newDf[col[i]] = df[col[i]].astype('category').cat.codes
        else: 
            newDf[col[i]] = df[col[i]]
    return newDf

###########################################################################################################

def standardize(df):
    # Standardization
    col = df.columns
    scale = preprocessing.StandardScaler().fit(df)
    df[col] = scale.transform(df[col]) 
    return df

###########################################################################################################

def normalize(df):
    col = df.columns
    norm = preprocessing.Normalizer().fit(df)
    df[col] = norm.transform(df[col]) 
    return df 

###########################################################################################################

def clense_data(df, stand = False, norm = False, cat = False, classification= None ): 
    if classification:
        new_df = df.loc[:, df.columns != classification ]
        temp = df.loc[:, df.columns == classification]
        temp1 = True
    else:
        new_df = df
        temp1 = False
        
    if cat:
        new_df = cat_to_useful(new_df)
    if stand: 
        new_df = standardize(new_df)
    if norm:
        new_df = normalize(new_df)
    if temp1: 
        return pd.concat([new_df,temp], axis=1)
    
    return new_df 

###########################################################################################################

def create_data_table(dfs, titles=None, spreadsheet_name = 'default', path = None):
    if path: 
        olddir = os.getcwd()
        os.chdir(path)
    dfs_dic = {'Sheet' + str(i): dfs[i] for i in range(len(dfs)) }

    # Title Feature will be added later 
    # if titles and len(titles) == len(dfs) :
    #     titles_check = titles
    # else:
    #     titles_check = ['Sheet' + str(i) for i in range(len(dfs))]
        
    writer = pd.ExcelWriter(spreadsheet_name + '.xlsx', engine='xlsxwriter')
    for df in dfs_dic.keys():
        # Write the dataframe data to XlsxWriter. Turn off the default header and
        # index and skip one row to allow us to insert a user defined header.
        dfs_dic[df].to_excel(writer, sheet_name =df, startrow=1, startcol=1, header=True, index=False)
        workbook = writer.book
        worksheet = writer.sheets[df]
        worksheet.set_column(1, dfs_dic[df].shape[1] , 12)
        merge_format = workbook.add_format({
        'bold': 1,
        'border': 1,
        'align': 'center'})


        # Merge 3 cells.
        worksheet.merge_range(0,1, 0, dfs_dic[df].shape[1], 'Add Title', merge_format)
    writer.save()
    os.chdir(olddir)
    return
        
###########################################################################################################