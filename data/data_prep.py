from sklearn.impute import KNNImputer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
import pandas as pd
import numpy as np
import os

N_SPLITS = 5
data_dir = "~/PycharmProjects/MTAttention/datasets"

def get_scores_for_imputer(imputer, X_missing, y_missing):
    estimator = make_pipeline(imputer, regressor)
    impute_scores = cross_val_score(estimator, X_missing, y_missing,
                                    scoring='neg_mean_squared_error',
                                    cv=N_SPLITS)
    return impute_scores


def get_impute_knn_score(X_missing, y_missing):
    imputer = KNNImputer(missing_values=np.nan, add_indicator=True)
    knn_impute_scores = get_scores_for_imputer(imputer, X_missing, y_missing)
    return knn_impute_scores.mean(), knn_impute_scores.std()


def prepare_data(site):

    path = os.path.join(data_dir, '{}_WQual_Level4_Imputed.csv'.format(site))
    df = pd.read_csv(path)
    '''data = pd.read_csv(path)
    dt = pd.Series(pd.to_datetime(data['Date']), name='DateTime')
    data = data.join(dt)
    del data['Date']
    data = data.drop(columns=["Absorbance_254", "UNH.ID..", "NPOC..mg.C.L.", "TDN..mg.N.L.", "Cl..mg.Cl.L.",
                              "NO3..mg.N.L.", "SO4..mg.S.L.", "Abs254", "SUVA", "Na..mg.Na.L.", "K..mg.K.L.",
                              "Mg..mg.Mg.L.", "Ca..mg.Ca.L.", "Closed.Cell.pH", "TSS..mg.L.", "PN", "NH4..ug.N.L.",
                              "DON", "PO4..ug.P.L.", "PC", "CO2ppm", "ABS254_SUNA", "DATETIME",
                              "Site", "DayID", "Year", "Month", "YearDay", "Hour", "Minute", "Second", "RECORD",
                              "Stage"])
    data_1 = data[['SpConductivity', 'Q', 'NO3_corrected', 'TempC', 'FDOMRFU', 'DateTime']]

    imputer = KNNImputer(n_neighbors=2, weights="uniform")
    data_trans = imputer.fit_transform(data_1.iloc[:, :-1])

    data_f = pd.DataFrame({'SpConductivity': data_trans[:, 0],
                           'Q': data_trans[:, 1],
                           'NO3_corrected': data_trans[:, 2],
                           'TempC': data_trans[:, 3],
                           'FDOMRFU': data_trans[:, 4],
                           'DateTime': data_1.iloc[:, 5]})

    df = pd.DataFrame(data_f)
    df.to_csv(os.path.join(data_dir, '{}_WQual_Level4_Imputed.csv'.format(site)))
    '''

    dt = pd.Series(pd.to_datetime(df['DateTime']), name='Date')
    df = df.join(dt)
    df = df.drop(columns=["Unnamed: 0"])
    df['diff'] = df['Date'].diff().dt.total_seconds().fillna(0).astype(int) > 10368000
    df['diff'] = df['diff'].apply(lambda x: 1 if x else 0)
    df_sub = []
    list_df_sub = []
    for i in range(len(df)):
        if df['diff'].iloc[i] == 0:
            df_sub.append(df.iloc[i])
        if df['diff'].iloc[i] == 1:
            list_df_sub.append(pd.DataFrame(df_sub))
            df_sub.clear()
            df_sub.append(df.iloc[i])
        if i == len(df) - 1:
            list_df_sub.append(pd.DataFrame(df_sub))
            df_sub.clear()

    for df in list_df_sub:
        df.set_index('DateTime', inplace=True)

    parent_dir = "split_ds/"
    #os.mkdir(parent_dir)
    i = 0
    for df in list_df_sub:
        path = os.path.join(parent_dir, '{}s_{}.csv'.format(site, i + 1))
        df.to_csv(path)
        i += 1


def main():
    prepare_data('BDC')


if __name__ == '__main__':
    main()
