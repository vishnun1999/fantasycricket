

import pandas as pd

import pickle
from point_prediction import ModelTrain, ModelPredict

from datetime import datetime
import pytz


def execute_model_train(datapath,modelname, predictors, cat_cols, target_col, usetimeseries=False):

    masterdf = pd.read_csv(datapath['featenggpath'])
    if usetimeseries:
        ts_prediction = ModelTrain.get_timeseries_forecast(masterdf, target_col, 'playername', 'ts_pred_points')
        masterdf = pd.merge(masterdf, ts_prediction, left_index=True, right_index=True, how='left')
        masterdf.to_csv(r'Data\time_series_output.csv', index=False)
        predictors = predictors + ['ts_pred_points']
    if modelname == 'movingaverage':
        return
    modeltrain = ModelTrain(masterdf, target_col, predictors, cat_cols, modelname)
    modeltrain.get_normalized_data()
    modeltrain.get_test_train(split_col='year', split_value=[2019])
    modelobjects = modeltrain.train_model(model=modelname)
    pickle.dump(modelobjects[2], open(datapath['modelpath'], 'wb'))
    pickle.dump(modelobjects[:2], open(datapath['encoderpath'], 'wb'))
    print(modeltrain.feat_imp_df)
    return


def execute_model_prediction(datapath,  predictors, modelname, cat_cols, pred_col, usetimeseries=False, predpath=False):


    if predpath:
        masterdf = pd.read_csv(datapath['predfeaturepath'])
    else:
        masterdf = pd.read_csv(datapath['featenggpath'])

    if modelname == 'rf':
        masterdf.fillna(-100, inplace=True)

    if usetimeseries:
        predictors = predictors + ['ts_pred_points']

    if modelname == 'movingaverage':
        masterdf[pred_col] = masterdf['total_points_playername_avg3']
        masterdf.to_csv(datapath['modelresultspath'], index=False)
        return masterdf

    modelpkl = pickle.load(open(datapath['modelpath'], 'rb'))
    enc = pickle.load(open(datapath['encoderpath'], 'rb'))
    mod_predict = ModelPredict(masterdf, enc, modelpkl, modelname, predictors, cat_cols, pred_col)
    mod_predict.get_normalized_data()
    masterdf[pred_col] = mod_predict.get_model_predictions()
    masterdf.to_csv(datapath['modelresultspath'], index=False)
    return masterdf





def get_team_details(datapath,index =0):

    tz_dubai = pytz.timezone('Asia/Dubai')
    datetime_dubai = datetime.now(tz_dubai)
    matchsummary = pd.read_csv(datapath['matchsummarypathipl20'])
    #matchid = matchsummary.iloc[next(x[0] for x in enumerate(pd.to_datetime(matchsummary['date']).tolist()) if x[1] > datetime_dubai), 0]
    matchid = 1216536

    today_match = matchsummary[matchsummary['matchid'] == matchid]
    print(today_match)
    team1 = today_match['team1'].iloc[index]
    team2 = today_match['team2'].iloc[index]
    venue = today_match['venue'].iloc[index].split(",")[index]
    return team1, team2, venue