def encode_categories(df):
    """
    Encodes the categorical features.
    Maps eng_wal and tenure to 1/0. One ho t encodes type, dropping "other".
    """
    import pandas as pd
    
    #Dummy variables for Type
    #Dropping Other column
    type_dummies = pd.get_dummies(df['Type'], prefix = 'Type', dtype=int)
    type_dummies.drop('Type_O', axis=1, inplace=True)
    
    df_numeric = df.join(type_dummies)
    df_numeric.drop('Type', axis=1, inplace=True)
    df_numeric['New'] = df_numeric['New'].map({'Y':1, 'N':0})

    #Mapping tenure and eng/wales
    df_numeric['Tenure'] = df_numeric['Tenure'].map({'F':1, 'L':0})
    df_numeric['Eng_Wal'] = df_numeric['Eng_Wal'].map({'England':1, 'Wales':0})

    return df_numeric

#####################################################################################

#Custom transformer for log of X parameters
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
class LogTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, log_income=False, log_listings=False):
        self.log_income = log_income
        self.log_listings = log_listings
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_output = X.copy()
        if self.log_income and ('Income' in X.columns.values):
            X_output['Income'] = np.log(X_output['Income'])
        if self.log_listings and ('listings_per_capita' in X.columns.values):
            X_output['listings_per_capita'] = np.log(X_output['listings_per_capita'])
        return X_output

# Custom target transformer for taking log of y and reverting fitted value to linear space
class ConditionalLogTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, log_target=False):
        self.log_target = log_target
    
    def fit(self, y):
        return self
    
    def transform(self, y):
        if self.log_target:
            return np.log(y)
        return y
    
    def inverse_transform(self, y):
        if self.log_target:
            return np.exp(y)
        return y

########################################################################################

#Function to plot fits and print r2
def assess_predict(y_real, y_predict, label=None, scale_limit=1000000):
    """
    Prints hexbin plots of the real and predicted prices
    """
    from matplotlib.ticker import FuncFormatter
    import matplotlib.pyplot as plt
    from sklearn.metrics import r2_score
    
    y_residual = y_real-y_predict
    
    def thousands_formatter(x, pos):
        return f'£{int(x/1000)}k'
    
    fig, axs = plt.subplots(1,3) 
    fig.set_size_inches(12, 4)
    fig.suptitle(label)
    
    axs[0].hexbin(y_real, y_predict, cmap='jet', extent=[0, scale_limit, 0, scale_limit])
    axs[0].plot([0,scale_limit],[0,scale_limit], c='black')
    axs[0].set_xlim(0,scale_limit)
    axs[0].set_ylim(0,scale_limit)
    axs[0].xaxis.set_major_formatter(FuncFormatter(thousands_formatter))
    axs[0].yaxis.set_major_formatter(FuncFormatter(thousands_formatter))
    axs[0].set_xlabel('True price')
    axs[0].set_ylabel('Predicted price')
    axs[0].set_title('Real and predicted prices')
    
    axs[1].hexbin(y_real, y_residual, cmap='jet', extent=[0,scale_limit, -scale_limit/2,scale_limit/2])
    axs[1].plot([0,scale_limit],[0,0], c='black')
    axs[1].set_xlim(0,scale_limit)
    axs[1].set_ylim(-scale_limit/2,scale_limit/2)
    axs[1].xaxis.set_major_formatter(FuncFormatter(thousands_formatter))
    axs[1].yaxis.set_major_formatter(FuncFormatter(thousands_formatter))
    axs[1].set_xlabel('True price')
    axs[1].set_ylabel('Residual')
    axs[1].set_title('Residual and true price')

    axs[2].hist(y_residual, bins=100, range=[-scale_limit,scale_limit])
    axs[2].xaxis.set_major_formatter(FuncFormatter(thousands_formatter))
    axs[2].set_xlabel('Residual')
    axs[2].set_ylabel('Count')
    axs[2].set_title('Residual histogram')
    
    fig.tight_layout()
    plt.show()

    r2 = r2_score(y_real, y_predict)
    print(f"R² = {r2:.3f}")

###################################################################################

#Feature importance test
def feature_importance_test(model_to_test, train_X, train_y, test_X, test_y, cpus=4):
    """
    Assesses the features included in the model to determine which are required
    Fits the training data using all the features, then determines the least import feature and removes it. 
    Fits training data again with the remaining features and removes the least important feature again
    Repeatus until only one feature remaining
    Calculates and prints the R2 values at each stage, so you can monitor how removing the features affects the quality of the fit
    Returns a pandas dataframe with the r2 values at each stage
    """
    import pandas as pd
    import numpy as np
    from sklearn.metrics import r2_score
    from sklearn.inspection import permutation_importance
    from sklearn.base import clone
    
    r2_scores = []
    r2_change=[]
    features = []
    previous_r_squared = None
    features_to_include = train_X.columns.values
    model = clone(model_to_test)

    #Iterate until only one feature remaining
    while len(features_to_include) > 0:
        #Train model and get r squared and improvement
        model.fit(train_X[features_to_include], train_y)
        r_squared = r2_score(test_y, model.predict(test_X[features_to_include]))
        if previous_r_squared is None:
            change = 0
        else:
            change = previous_r_squared-r_squared
        r2_change.append(change)
        previous_r_squared = r_squared
        r2_scores.append(r_squared)
    
        #Print r2 and features
        print(f'r2 = {r_squared:.3f}, change = {change:.3f}')
        print(f'features: {features_to_include}')
    
        #Calculate feature importance
        imp = permutation_importance(model, train_X[features_to_include],train_y, n_jobs=cpus, random_state=89)
    
        #Determine least important feature
        least_important_feature_index = imp.importances_mean.argsort()[0]
        least_important_feature = features_to_include[least_important_feature_index]
        features.append(least_important_feature)
        print(f'least important feature: {least_important_feature}')
        print()
        
        #remove least important feature from the features list
        features_to_include = np.delete(features_to_include, least_important_feature_index)

    return pd.DataFrame({'least_important_feature': features, 'r2':r2_scores, 'r2_change':r2_change})

###################################################################################

def results_within_limit(true_price, predicted_price, limit):
    """
    Calculates and returns the percentage of predicted results wich are within x% (limit) of the true result
    """
    abs_pc_residual = abs((true_price-predicted_price)/true_price)
    pc_within_limit = abs_pc_residual.loc[lambda x : x <= limit].count()/true_price.count()
    return pc_within_limit
