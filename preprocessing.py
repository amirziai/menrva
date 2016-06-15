import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier


def remove_columns(df, exclusions):
    if len(exclusions) > 0:
        return df[df.columns.difference(exclusions)]
    else:
        return df
    
    
def report_column_alteration(column, action, notes):
    if type(notes) == float:
        notes = '{:0.1f}%'.format(notes)
        
    print '%15s %25s %15s' % (column, action, notes)

    
def prepare_df(df, target, target_label_encoder=None, report=True):
    columns_to_ohe = []  # columns to be one-hot-encoded
    columns_to_remove = []
    target_label_encoder = None
    
    for col, col_type in zip(df.dtypes.index, df.dtypes.values):
        if col_type == 'O':
            if col == target:
                if report:
                    print('Encoding the target variable')
                    
                if not target_label_encoder:
                    target_label_encoder = LabelEncoder()
                    target_label_encoder.fit(df[col])
                
                df[col] = target_label_encoder.transform(df[col])

            else:
                ratio = df[col].nunique() / float(len(df))
                
                if ratio < 0.1:
                    columns_to_ohe.append(col)
                else:
                    columns_to_remove.append(col)
                    
                    if report:
                        report_column_alteration(col, 'Removed- overly unique', ratio * 100)

        else:
            count_na = int(len(df) - df[col].count())
            if count_na > 0:
                df[col] = df[col].fillna(-999)
                
                if report:
                    report_column_alteration(col, 'Fill NA', (100 * count_na / float(len(df))))
            
    return df, columns_to_remove, columns_to_ohe, target_label_encoder


def prepare_df_pipeline(df, target, exclusions, target_label_encoder=None, report=True):
    df = remove_columns(df, exclusions)
    df, columns_to_remove, columns_to_ohe, target_label_encoder = prepare_df(df,
                                                                             target,
                                                                             target_label_encoder,
                                                                             report)
    df = remove_columns(df, columns_to_remove)
    df = pd.get_dummies(df, columns=columns_to_ohe)
    X, y = get_xy(df, target)
    
    return X, y, target_label_encoder


def get_xy(df, target):
    X = df[df.columns.difference([target])]
    y = df[target]
    return X, y


def train_test_xy(df, target, exclusions, file_name_test, test_set_percentage):
    X, y, target_label_encoder = prepare_df_pipeline(df, target, exclusions)

    if file_name_test:
        X_train, y_train = X, y
        df_test = pd.read_csv(file_name_test)
        X_test, y_test, _ = prepare_df_pipeline(df_test, target, exclusions, target_label_encoder, False)

    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_set_percentage)
        
    print '=' * 30
    print 'Features  : %s' % X_train.shape[1]
    print 'Train set : %s' % X_train.shape[0]
    print 'Test set  : %s' % X_test.shape[0]
    print '=' * 30
        
    return X_train, X_test, y_train, y_test, target_label_encoder