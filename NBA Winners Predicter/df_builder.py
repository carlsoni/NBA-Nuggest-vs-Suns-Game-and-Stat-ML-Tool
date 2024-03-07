import pandas as pd
import numpy as np

def generate_stats(player, df, features, mode):
    player_df = df[df['PLAYER_NAME'] == player]
    stats = features
    means = player_df[stats].mean()
    stds = player_df[stats].std()

    new_row = {}

    # Generate stats within two std below the mean and one std above the mean
    for stat in stats:
        mean = means[stat]
        std = stds[stat]
        # Calculate the lower and upper bounds
        lower_bound = mean - mode * std
        upper_bound = mean + std
        # Generate a random value within the bounds
        random_value = np.random.uniform(lower_bound, upper_bound)
        # Assign the generated value to the stat in the new row
        new_row[stat] = random_value

    new_row['PLAYER_NAME'] = player

    return new_row

