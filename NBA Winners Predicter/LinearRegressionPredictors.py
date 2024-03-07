import pandas as pd
import numpy as np
import df_builder as dfb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings

# Suppress future warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

nuggets_df = pd.read_csv('nuggets_game_logs_2023_24.csv')
suns_df = pd.read_csv('suns_game_logs_2023_24.csv')

df = pd.concat([nuggets_df, suns_df], ignore_index=True)

# feature selection
features = ['MIN', 'FGA', 'FG3A', 'FTA', 'OREB', 'DREB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PLUS_MINUS']
X = df[features]
y = df[['PTS', 'PLAYER_NAME']]  # Target variable

# Split the dataset
X_train, X_test, y_train, y_test_with_names = train_test_split(X, y, test_size=0.2, random_state=42)
y_train, player_names_test = y_train['PTS'], y_test_with_names['PLAYER_NAME']
y_test = y_test_with_names['PTS']


# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and evaluate
predictions = model.predict(X_test)
predictions = np.clip(predictions, a_min=0, a_max=None)
predictions = np.round(predictions, decimals=0)
# Calculate and print RMSE for the numeric predictions vs actual
mse = mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mse)
print("-------------------------------------Inital Model Trained On Overall Stats To predict PTS---------------------------------")
print(f"\n\n\nMean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")
print(f"Root Mean Absolute Error: {rmse}")

#print("\n\n\n------------------------------RUNNING GAME PREDICTIONS-------------------------------------")
mode = 1.2
mode1 = 1.2
outcome = []
nugget_scores = []
suns_scores = []
nuggets_starting_lineup = {
    'Christian Braun', 
    'Michael Porter Jr', 
    'Kentavious Caldwell-Pope',
    'Reggie Jackson',
    'Peyton Watson',
    'Nikola Jokic',
    'Zeke Nnaji',
    'Jamal Murray',
    'Aaron Gordon'
}
suns_starting_lineup = {
    "Royce O'Neale",
    'Bradley Beal',
    'Grayson Allen',
    'Bol Bol',
    'Drew Eubanks',
    'Jusuf Nurkic',
    'David Roddy',
    'Eric Gordon',
    'Kevin Durant',
    'Saben Lee'
}
for i in range(1, 101):
    
    given_game = pd.DataFrame(columns=features + ['PLAYER_NAME'])
    for nugget in nuggets_starting_lineup:
        new_row = dfb.generate_stats(nugget, df, features, mode1) 
        given_game = given_game._append(new_row, ignore_index=True)

    X_given_game = given_game[features]

    # Generate predictions
    given_game_predictions = model.predict(X_given_game)
    given_game_predictions = np.clip(given_game_predictions, a_min=0, a_max=None)
    given_game_predictions = np.round(given_game_predictions, decimals=0)

    # Add predictions to the given_game DataFrame
    given_game['Predicted_PTS'] = given_game_predictions

    suns_game = pd.DataFrame(columns=features + ["PLAYER_NAME"])
    for bitch in suns_starting_lineup:
        new_row = dfb.generate_stats(bitch, df, features, mode)
        suns_game = suns_game._append(new_row, ignore_index=True)
    
    x_suns_game = suns_game[features]

    # generate suns predictions
    suns_game_predictions = model.predict(x_suns_game)
    suns_game_predictions = np.clip(suns_game_predictions, a_min=0, a_max=None)
    suns_game_predictions = np.round(suns_game_predictions, decimals=0)

    # Add predicted to suns game
    suns_game['Predicted_PTS'] = suns_game_predictions

    # Display stats organized by player
    #for index, row in suns_game.iterrows():
    #    print(f"{row['PLAYER_NAME']} : {int(row['Predicted_PTS'])} PTS")
    nuggets_score = sum(given_game['Predicted_PTS'])
    suns_score = sum(suns_game['Predicted_PTS'])
    nugget_scores.append(nuggets_score)
    suns_scores.append(suns_score)
    
    if nuggets_score > suns_score:
        #print(f"Nuggets Win!!!! {nuggets_score} to {suns_score}")
        outcome.append(1)
    else:
        #print(f"Suns Win!!!! {suns_score} to {nuggets_score}")
        outcome.append(0)

nugget_wins = outcome.count(1)
suns_wins = outcome.count(0)

print("\n\n\n--------------------------------Number of Games Won Across 100 Games--------------------------------------------------")
print(f"The Nuggets won {nugget_wins} out of 100 Test")
print(f"The Suns won {suns_wins} out of 100 games")
print("--------------------------------Average Score Across 100 Games--------------------------------------------------")
print(f"The average Nuggets Score is {np.round(np.mean(nugget_scores), decimals=0)} and the ground truth was {107} with a difference of {abs(np.round(np.mean(nugget_scores)) - 107)}")
print(f"The average Suns Score is {np.round(np.mean(suns_scores), decimals=0)} and the ground truth was {117} with a difference of {abs(np.round(np.mean(suns_scores), decimals=0) - 117)}")

# Train moew models to detirmine reb, assists, and PTS by player based on min played 
print("\n\n\n--------------------------------The Rest of The program is based off of Min Played to Detirmine Stats and Final Score--------------------------------------------------")
X = df[['MIN']]  # Features
y_pts = df['PTS']  # Target variable for Points
y_reb = df['REB']  # Target variable for Rebounds
y_ast = df['AST']  # Target variable for Assists

# Split the dataset for each target
X_train_pts, X_test_pts, y_train_pts, y_test_pts = train_test_split(X, y_pts, test_size=0.2, random_state=42)
X_train_reb, X_test_reb, y_train_reb, y_test_reb = train_test_split(X, y_reb, test_size=0.2, random_state=42)
X_train_ast, X_test_ast, y_train_ast, y_test_ast = train_test_split(X, y_ast, test_size=0.2, random_state=42)

# Initialize the models
model_pts = LinearRegression()
model_reb = LinearRegression()
model_ast = LinearRegression()

# Train the models
model_pts.fit(X_train_pts, y_train_pts)
model_reb.fit(X_train_reb, y_train_reb)
model_ast.fit(X_train_ast, y_train_ast)

# Predict with the models
predictions_pts = model_pts.predict(X_test_pts)
predictions_reb = model_reb.predict(X_test_reb)
predictions_ast = model_ast.predict(X_test_ast)

# Ensure predictions are within a sensible range
predictions_pts = np.clip(predictions_pts, a_min=0, a_max=None)
predictions_reb = np.clip(predictions_reb, a_min=0, a_max=None)
predictions_ast = np.clip(predictions_ast, a_min=0, a_max=None)

# Round predictions to the nearest whole number
predictions_pts = np.round(predictions_pts, decimals=0)
predictions_reb = np.round(predictions_reb, decimals=0)
predictions_ast = np.round(predictions_ast, decimals=0)

# Evaluate each model
mse_pts = mean_squared_error(y_test_pts, predictions_pts)
mse_reb = mean_squared_error(y_test_reb, predictions_reb)
mse_ast = mean_squared_error(y_test_ast, predictions_ast)

mae_pts = mean_absolute_error(y_test_pts, predictions_pts)
mae_reb = mean_absolute_error(y_test_reb, predictions_reb)
mae_ast = mean_absolute_error(y_test_ast, predictions_ast)

rmse_pts = np.sqrt(mse_pts)
rmse_reb = np.sqrt(mse_reb)
rmse_ast = np.sqrt(mse_ast)

# Print evaluation metrics
print("Evaluation Metrics for PTS Prediction:")
print(f"Mean Squared Error: {mse_pts}")
print(f"Mean Absolute Error: {mae_pts}")
print(f"Root Mean Squared Error: {rmse_pts}\n")

print("Evaluation Metrics for REB Prediction:")
print(f"Mean Squared Error: {mse_reb}")
print(f"Mean Absolute Error: {mae_reb}")
print(f"Root Mean Squared Error: {rmse_reb}\n")

print("Evaluation Metrics for AST Prediction:")
print(f"Mean Squared Error: {mse_ast}")
print(f"Mean Absolute Error: {mae_ast}")
print(f"Root Mean Squared Error: {rmse_ast}\n")

nuggets_starting_lineup_and_min_dict = {
    'Christian Braun' : 19,                        
    'Michael Porter Jr' : 40, 
    'Kentavious Caldwell-Pope' : 38,
    'Reggie Jackson' : 16,
    'Peyton Watson' : 23,
    'Nikola Jokic' : 44,
    'Zeke Nnaji' : 9,
    'Jamal Murray' : 42,
    'Aaron Gordon' : 34
}
suns_starting_lineup_and_min_dict = {
    "Royce O'Neale" : 39,
    'Bradley Beal' : 36,
    'Grayson Allen' : 41,
    'Bol Bol' : 13,
    'Drew Eubanks' : 23,
    'Jusuf Nurkic' : 30,
    'David Roddy' : 5,
    'Eric Gordon' : 22,
    'Kevin Durant' : 44,
    'Saben Lee' : 13
}
nuggest_prediction_matrix = []
nuggets_PTS_predictions = []
nuggets_AST_predictions = []
nuggets_REB_predictions = []
print("------------------------------Nuggets Individual Stat Predictions----------------------------------")
print(f"{'Player':<25}{'PTS':>5}{'AST':>5}{'REB':>5}")
for nugget in nuggets_starting_lineup_and_min_dict:

    # get the min a given player was in 
    minutes = np.array([nuggets_starting_lineup_and_min_dict[nugget]]).reshape(1, -1)

    # predict PTS, AST, and REB
    predicted_points = np.round(model_pts.predict(minutes), decimals=0)
    predicted_ast = np.round(model_ast.predict(minutes), decimals=0)
    predicted_rebs = np.round(model_reb.predict(minutes), decimals=0)

    # print predictions for given player
    print(f"{nugget:<25}{predicted_points[0]:>5}{predicted_ast[0]:>5}{predicted_rebs[0]:>5}")

    # append stats for future traking
    nuggets_PTS_predictions.append(predicted_points[0])
    nuggets_AST_predictions.append(predicted_ast[0])
    nuggets_REB_predictions.append(predicted_rebs[0])

nuggest_prediction_matrix.append(nuggets_PTS_predictions)
nuggest_prediction_matrix.append(nuggets_AST_predictions)
nuggest_prediction_matrix.append(nuggets_REB_predictions)

suns_prediction_matrix = []
suns_PTS_predictions = []
suns_AST_predictions = []
suns_REB_predictions = []
print("\n\n\n------------------------------Suns Individual Stat Predictions----------------------------------")
print(f"{'Player':<25}{'PTS':>5}{'AST':>5}{'REB':>5}")
for bitch in suns_starting_lineup_and_min_dict:

    # grab min for given player
    minutes = np.array([suns_starting_lineup_and_min_dict[bitch]]).reshape(1, -1)

    # predict and ajust stats
    predicted_points = np.clip(np.round(model_pts.predict(minutes), decimals=0), a_min=0, a_max=None)
    predicted_ast = np.clip(np.round(model_ast.predict(minutes), decimals=0), a_max=None, a_min=0)
    predicted_rebs = np.round(model_reb.predict(minutes), decimals=0)

    # print stat predictions
    print(f"{bitch:<25}{predicted_points[0]:>5}{predicted_ast[0]:>5}{predicted_rebs[0]:>5}")

    # append stat predictions
    suns_PTS_predictions.append(predicted_points[0])
    suns_AST_predictions.append(predicted_ast[0])
    suns_REB_predictions.append(predicted_rebs[0])

suns_prediction_matrix.append(suns_PTS_predictions)
suns_prediction_matrix.append(suns_AST_predictions)
suns_prediction_matrix.append(suns_REB_predictions)

# ground truth stats grabed via google
nuggets_ground_truth_matrix = []
nuggets_PTS_ground_truth = [7, 20, 11, 3, 4, 25, 2, 28, 7]
nuggets_AST_ground_truth = [1, 0, 3, 0, 4, 5, 0, 9, 3]
nuggets_REB_ground_truth = [4, 7, 5, 2, 4, 16, 3, 7, 3]

nuggets_ground_truth_matrix.append(nuggets_PTS_ground_truth)
nuggets_ground_truth_matrix.append(nuggets_AST_ground_truth)
nuggets_ground_truth_matrix.append(nuggets_REB_ground_truth)


suns_ground_truth_matrix = []
suns_PTS_ground_truth = [8, 16, 28, 2, 10, 7, 3, 4, 35, 4]
suns_AST_ground_truth = [6, 6, 4, 1, 0, 6, 0, 2, 5, 2]
suns_REB_ground_truth = [5, 6, 5, 4, 8, 12, 0, 1, 8, 3]

suns_ground_truth_matrix.append(suns_PTS_ground_truth)
suns_ground_truth_matrix.append(suns_AST_ground_truth)
suns_ground_truth_matrix.append(suns_REB_ground_truth)


print("\n\n\n-----------------Nuggets Individual Stat Predictions Evaluation-----------------------------")
stats = ['PTS', 'AST', 'REB']
for i, stat_predictions in enumerate([nuggets_PTS_predictions, nuggets_AST_predictions, nuggets_REB_predictions]):
    ground_truth = nuggets_ground_truth_matrix[i]
    mse = mean_squared_error(ground_truth, stat_predictions)
    mae = mean_absolute_error(ground_truth, stat_predictions)
    rmse = np.sqrt(mse)
    print(f"\n{stats[i]}:")
    print(f"Mean Squared Error: {round(mse, 2)}")
    print(f"Mean Absolute Error: {round(mae, 2)}")
    print(f"Root Mean Squared Error: {round(rmse,2)}")


nuggets_prediction_matrix = np.array(nuggest_prediction_matrix)
nuggets_ground_truth_matrix = np.array(nuggets_ground_truth_matrix)

mse_matrix = mean_squared_error(nuggets_ground_truth_matrix, nuggets_prediction_matrix)
mae_matrix = mean_absolute_error(nuggets_ground_truth_matrix, nuggets_prediction_matrix)
rmse_matrix = np.sqrt(mse_matrix)

print("\n--------------------Nuggets Overall Team Predictions Evaluation------------------------------")
print(f"Matrix Mean Squared Error: {round(mse_matrix, 2)}")
print(f"Matrix Mean Absolute Error: {round(mae_matrix, 2)}")
print(f"Matrix Root Mean Squared Error: {round(rmse_matrix, 2)}")


print("\n\n--------------------------------Suns Individual Stat Predictions Evaluation----------------------------------------")
for i, stat_predictions in enumerate([suns_PTS_predictions, suns_AST_predictions, suns_REB_predictions]):
    ground_truth = suns_ground_truth_matrix[i]
    mse = mean_squared_error(ground_truth, stat_predictions)
    mae = mean_absolute_error(ground_truth, stat_predictions)
    rmse = np.sqrt(mse)
    print(f"\n{stats[i]}:")
    print(f"Mean Squared Error: {round(mse, 2)}")
    print(f"Mean Absolute Error: {round(mae, 2)}")
    print(f"Root Mean Squared Error: {round(rmse, 2)}")


suns_prediction_matrix = np.array(suns_prediction_matrix)
suns_ground_truth_matrix = np.array(suns_ground_truth_matrix)

mse_matrix = mean_squared_error(suns_ground_truth_matrix, suns_prediction_matrix)
mae_matrix = mean_absolute_error(suns_ground_truth_matrix, suns_prediction_matrix)
rmse_matrix = np.sqrt(mse_matrix)

print("\n------------------------------Suns Overall Team Predictions Evaluation--------------------------------")
print(f"Matrix Mean Squared Error: {round(mse_matrix, 2)}")
print(f"Matrix Mean Absolute Error: {round(mae_matrix, 2)}")
print(f"Matrix Root Mean Squared Error: {round(rmse_matrix, 2)}")

print("\n\n------------------------------Final Scores-----------------------------\n")
print(f"The Nuggets Were Predicted to score: {sum(nuggets_PTS_predictions)} when in reality they Scored: {107} making the difference {abs(sum(nuggets_PTS_predictions) - 107)}")
print(f"The Suns Were Predicted to score: {sum(suns_PTS_predictions)} when in reality they Scored: {117} making the difference {abs(sum(suns_PTS_predictions) - 117)}")

print("\n The Game Ended Up in the Suns Favor Being 117-107 in overtime showing how closely matched the Nuggets are to the Suns \n With most test being within a few point difference.\n\n")
