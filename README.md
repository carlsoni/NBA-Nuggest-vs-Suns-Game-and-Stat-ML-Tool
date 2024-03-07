# NBA Game Prediction Project README

## Overview

This project aims to predict the outcomes of NBA games with a focus on matchups between the Denver Nuggets and the Phoenix Suns for the 2023-24 season. It utilizes machine learning models to forecast player performances and game scores based on historical statistical data.

## Getting Started

To run this project, you will need Python 3.8 or later and the following libraries:

- pandas
- numpy
- scikit-learn
- nba_api

These can be installed via pip:

```bash
pip install pandas numpy scikit-learn nba_api
```

## Project Structure

- `nuggets_game_logs_2023_24.csv` - Game logs for the Denver Nuggets players for the 2023-24 season.
- `suns_game_logs_2023_24.csv` - Game logs for the Phoenix Suns players for the 2023-24 season.
- `df_builder.py` - Contains functions to generate predicted statistics for players.
- `LinearRegressionPredictors.py` - The main script where the model training and simulations are executed.
- `data_collector.py` - Script used to generate csv files from the nba_api

## Usage

1. **Prepare the Data:** Ensure the game logs for the Nuggets and Suns are updated and placed in the same directory as the scripts.
2. **Run the Model:** Execute the `model.py` script to start the simulation and predictions.

```bash
python LinearRegressionPredictors.py
```

3. **View Results:** The script will output the game predictions, including individual player performances and team scores, to the console.

## Features

- **Data Preparation:** Combines game logs from both teams into a single dataset for analysis.
- **Feature Selection:** Identifies key statistics relevant to predicting player performance.
- **Model Training:** Uses Linear Regression to forecast points, assists, and rebounds based on selected features.
- **Simulation:** Conducts 100 simulated matchups between the Nuggets and Suns to predict outcomes and scores.
- **Evaluation:** Assesses the accuracy of predictions using MSE, MAE, and RMSE.

## Customizing the Project

You can customize the project by adjusting the features selected for prediction or by training the models on different sets of data. For instance, you might incorporate more player statistics or use data from additional seasons to refine the predictions.

## Contributions

Contributions to this project are welcome. Please feel free to fork the repository, make changes, and submit pull requests with improvements or bug fixes.

## License

This project is open-sourced under the MIT License. See the LICENSE file for more details.

## Acknowledgments

Thanks to the creators of the `nba_api` library for providing an easy way to access NBA statistics and data, enabling this and many other sports analytics projects.
