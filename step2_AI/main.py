import neat
import visualize
import pandas as pd
#import pandas_ta as ta
import os
import pickle
import math
#from numba import jit, cuda
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as patches
import random
import colorsys
        
import json
json_filename = "ai_input.json"

# Load the JSON data from the file
with open(json_filename, 'r') as json_file:
    small_arrays_as_lists = json.load(json_file)

# Convert the lists back to NumPy arrays
train_data = [np.array(arr) for arr in small_arrays_as_lists]

def random_bright_color():
    h = random.uniform(0, 1)  # Random hue
    s = random.uniform(0.7, 1)  # Higher saturation for brighter colors
    v = random.uniform(0.7, 1)  # Higher value for brighter colors
    return colorsys.hsv_to_rgb(h, s, v)

def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


class SIM:
    def __init__(self, df):
        self.df = df

    def calculate_percentage_change(self, last_7_rows):
        # Calculate the differences from row to row
        arrName = ['Open', 'High', 'Low', 'Close']
        output = []

        for column_name in arrName:
            percentage_change = last_7_rows[column_name].pct_change().tolist()
            percentage_change = [round(value, 5) for value in percentage_change]
            percentage_change = percentage_change[1:]

            for item in percentage_change:
                output.append(item)

        binary_signal = [1 if close > open else 0 for close, open in zip(last_7_rows['Close'], last_7_rows['Open'])]
        binary_signal[1:]
        for item in binary_signal:
                output.append(item)
                
        return output

    def test_ai(self, genome, config):
        net = neat.nn.FeedForwardNetwork.create(genome, config)

        best_result = 0
        array1_final = []
        array2_final = []
        for i in range(20):
            result = 0
            random_index = np.random.randint(0, len(self.df))
            random_array = self.df[random_index]
            formatedInput = []
            for item in random_array:
                for individual_item in item:
                    formatedInput.append(individual_item)

            output = net.activate(( formatedInput ))
            threshold = 0.8
            binary_data = [1 if num >= threshold else 0 for num in output]
            store_locations = np.array(binary_data).reshape(5, 5)
            formatedInput = np.array(formatedInput).reshape(5, 5)

            array1 = formatedInput
            array2 = store_locations

            # Iterate through both arrays
            for i in range(5):
                for j in range(5):
                    if array1[i, j] == 1:
                        neighbor_indices = [(i - 1, j - 1), (i - 1, j), (i - 1, j + 1),
                                            (i, j - 1), (i, j + 1),
                                            (i + 1, j - 1), (i + 1, j), (i + 1, j + 1)]
                        
                        # Count the number of 1 values in the neighbors from the second array
                        count_ones = sum(array2[n] for n in neighbor_indices if 0 <= n[0] < 5 and 0 <= n[1] < 5)
                        
                        if count_ones == 1:
                            result += 10
                        elif count_ones > 1:
                            result -= 5 * count_ones

                    if array2[i, j] == 1:
                        neighbor_indices = [(i - 1, j - 1), (i - 1, j), (i - 1, j + 1),
                                            (i, j - 1), (i, j + 1),
                                            (i + 1, j - 1), (i + 1, j), (i + 1, j + 1), (i, j)]
                        
                        # Check if every neighboring value in array1 is zero
                        all_zeros_in_array1 = all(array1[n] == 0 for n in neighbor_indices if 0 <= n[0] < 5 and 0 <= n[1] < 5)
                        
                        if all_zeros_in_array1:
                            result -= 150
                        else:
                            for n in neighbor_indices:
                                if 0 <= n[0] < 5 and 0 <= n[1] < 5:
                                    if array1[n] == 0:
                                        result -= 3
                                    elif array1[n] == 1:
                                        result += 10


            if result > best_result:
                best_result = result
                array1_final = array1
                array2_final = array2

        print(array1_final)
        print(array2_final)

        # Create a custom colormap with inverted colors
        cmap = ListedColormap(['white', 'black'])

        # Create subplots
        fig, ax = plt.subplots()

        # Display array 1 with the custom colormap
        ax.imshow(array1_final, cmap=cmap)

        # Overlay red circles from array 2 with random colors for circles and boxes
        for i in range(array2_final.shape[0]):
            for j in range(array2_final.shape[1]):
                if array2[i, j] == 1:
                    color = random_bright_color()
                    circle = plt.Circle((j, i), 0.4, color=color, fill=True)
                    ax.add_patch(circle)
                    
                    box = patches.Rectangle((j - 1.45, i - 1.45), 2.9, 2.9, linewidth=1, edgecolor=color, fill=False, linestyle='dashed')
                    ax.add_patch(box)

        ax.set_title('Overlay of Array 2 with Random Colors')

        # Set aspect ratio to equal
        ax.set_aspect('equal')

        # Show the plot
        plt.show()
        
   
    def train_ai(self, genome, config):
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        
        run = True
        # Variable to store the final result
        result = 0
        
        while run:
            for i in range(50):
                random_index = np.random.randint(0, len(self.df))
                random_array = self.df[random_index]
                formatedInput = []
                for item in random_array:
                    for individual_item in item:
                        formatedInput.append(individual_item)

                output = net.activate(( formatedInput ))
                threshold = 0.8
                binary_data = [1 if num >= threshold else 0 for num in output]
                store_locations = np.array(binary_data).reshape(5, 5)
                formatedInput = np.array(formatedInput).reshape(5, 5)

                array1 = formatedInput
                array2 = store_locations

                # Iterate through the first array
                for i in range(5):
                    for j in range(5):
                        if array1[i, j] == 1:
                            neighbor_indices = [(i - 1, j - 1), (i - 1, j), (i - 1, j + 1),
                                                (i, j - 1), (i, j + 1),
                                                (i + 1, j - 1), (i + 1, j), (i + 1, j + 1)]
                            
                            # Count the number of 1 values in the neighbors from the second array
                            count_ones = sum(array2[n] for n in neighbor_indices if 0 <= n[0] < 5 and 0 <= n[1] < 5)
                            
                            if count_ones == 1:
                                result += 10
                            elif count_ones > 1:
                                result -= 5 * count_ones

            break
        
        genome.fitness = result/50 + 0.001


def eval_genomes(genomes, config):
    for i, (genome_id, genome) in enumerate(genomes):
        game = SIM(train_data)
        game.train_ai(genome, config)





def run_neat(config):
    #p = neat.Checkpointer.restore_checkpoint('./neat-checkpoint-0')
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(1))
    
    winner = p.run(eval_genomes, 1000)

    visualize.draw_net(config, winner, view=True, filename="NET.gv", show_disabled=True)
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)
    
    with open("best.pickle", "wb") as f:
        pickle.dump(winner, f)
        print('SAVED THE PICKLE')






def test_ai(config, df_test):
    with open("best.pickle", "rb") as f:
        winner = pickle.load(f)
        print('LOADED THE PICKLE')

    #visualize.draw_net(config, winner, view=True, filename="NET.gv", show_disabled=True)

    print(winner.size())
    print(winner)
    game = SIM(df_test)
    game.test_ai(winner, config)




if __name__ == "__main__":
   
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config.txt")

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)    

    #run_neat(config)
    test_ai(config, train_data)
