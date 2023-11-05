from uszipcode import SearchEngine
import pandas as pd
import numpy as np
import json


file_path = "formatedData/zip_code_list.csv"
df_zips = pd.read_csv(file_path)
firstTeeZips = []

for index, row in df_zips.iterrows():
    first_column_value = row[1]
    firstTeeZips.append(first_column_value)

search = SearchEngine()
result = search.by_median_household_income(lower=0, upper=999999999)
sorted_zipcodes = search.by_population(sort_by='median_household_income', ascending=True, returns=100000)

zipCodes = []

for item in sorted_zipcodes:
    if(item.median_household_income > 30000):
        break
    zipCodes.append(item.zipcode)

zip_codes_to_check = zipCodes
zip_codes_to_remove = firstTeeZips

filtered_zip_codes = [zip_code for zip_code in zip_codes_to_check if zip_code not in zip_codes_to_remove]
print(len(filtered_zip_codes))

neighbor_zipcodes_dict = {}

# Iterate through each zip code in your list
for item in filtered_zip_codes:
    z = search.by_zipcode(str(item))
    result = search.by_coordinates(float(z.lat), float(z.lng), radius=15, returns=100000)
    zipcodes = [zipcodeInfo.zipcode for zipcodeInfo in result]

    # Filter neighboring zip codes to keep only those in filtered_zip_codes
    zipcodes = [zipcode for zipcode in zipcodes if zipcode in filtered_zip_codes]

    # Add the current zip code and its neighbors to the dictionary
    neighbor_zipcodes_dict[str(item)] = zipcodes

# Create a list of lists to store the 2D array
neighbor_zipcodes_2d = []

# Function to find the group of connected zip codes
def find_group(zipcode):
    group = []
    visited = set()
    
    def dfs(current):
        visited.add(current)
        group.append(current)
        for neighbor in neighbor_zipcodes_dict[current]:
            if neighbor not in visited:
                dfs(neighbor)
    
    dfs(zipcode)
    return group

# Iterate through zip codes and build groups
visited = set()
for zipcode in filtered_zip_codes:
    if zipcode not in visited:
        group = find_group(zipcode)
        visited.update(group)
        neighbor_zipcodes_2d.append(group)

# Print the 2D list
#for group in neighbor_zipcodes_2d:
    #print(group)
print("------")
formated_arrays = []

for index_of_zipcode_groups in range(len(neighbor_zipcodes_2d)):
    rectangle_data = []
    for item in neighbor_zipcodes_2d[index_of_zipcode_groups]:
        z = search.by_zipcode(str(item))

        width = float(z.bounds_east) - float(z.bounds_west)
        height = float(z.bounds_north) - float(z.bounds_south)
        rectangle_data.append([round(float(z.bounds_south) - width/2, 5), round(float(z.bounds_west) - height/2, 5), round(width, 5), round(height, 5)])


    x_coords = []
    y_coords = []

    for item in rectangle_data:
        x_coords.append(item[0])
        y_coords.append(item[1])
        
    # Find the minimum x and y values
    min_x = min(x_coords)
    min_y = min(y_coords)

    # Apply the transformation
    transformed_x_coords = [x - min_x for x in x_coords]
    transformed_y_coords = [y - min_y for y in y_coords]

    for i in range(len(rectangle_data)):
        rectangle_data[i][0] = round(transformed_x_coords[i], 5)
        rectangle_data[i][1] = round(transformed_y_coords[i], 5)
        

    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots()

    min_x = 100000
    min_y = 100000
    max_x = -100000
    max_y = -100000
    for x, y, width, height in rectangle_data:
        if x < min_x: min_x = x
        if y < min_y: min_y = y
        if x+width > max_x: max_x = x+width
        if y+height > max_y: max_y = y+height
        
        #print(str(x) + " | " + str(y))
        rect = Rectangle((x, y), width, height, linewidth=1, edgecolor='black', facecolor='black', fill=True)
        ax.add_patch(rect)

    ax.set_aspect('equal')
    # Set axis limits if needed
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)

    plt.show()
    plt.close()


    multiplier = 68.9 / 3

    min_x *= multiplier
    min_y *= multiplier
    max_x *= multiplier
    max_y *= multiplier

    min_x = int(min_x)
    min_y = int(min_y)
    max_x = int(max_x)
    max_y = int(max_y)

    # Create an empty grid with zeros
    grid_width = max_x - min_x + 1
    grid_height = max_y - min_y + 1
    grid = np.zeros((grid_height, grid_width), dtype=int)

    # Fill the grid with 1s where rectangles exist
    for x, y, width, height in rectangle_data:
        x = int(x*multiplier)
        y = int(y*multiplier)
        width = int(width*multiplier)
        height = int(height*multiplier)
        
        x1 = int((x - min_x))  # Scale to integer
        y1 = int((y - min_y))  # Scale to integer
        x2 = x1 + int(width)  # Scale to integer
        y2 = y1 + int(height)  # Scale to integer
        grid[y1:y2, x1:x2] = 1


    num_rows = grid.shape[0] // 5
    num_cols = grid.shape[1] // 5

    # Loop through the big array and extract 5x5 slices without overlapping
    for i in range(num_rows):
        for j in range(num_cols):
            new_array = grid[i * 5:(i + 1) * 5, j * 5:(j + 1) * 5]
            
            # Check if the extracted slice is 5x5
            if new_array.shape == (5, 5):
                formated_arrays.append(new_array)

unique_arrays = set()
filtered_main_array = []

for array in formated_arrays:
    array_list = array.tolist()
    array_tuple = tuple(tuple(row) for row in array_list)
    if array_tuple not in unique_arrays:
        filtered_main_array.append(array_list)
        unique_arrays.add(array_tuple)

formated_arrays = filtered_main_array

print("Saving")
print(len(formated_arrays))

#small_arrays_as_lists = [small_array.tolist() for small_array in formated_arrays]

# Define the filename for the JSON file
json_filename = "ai_input.json"

# Save the list to a JSON file
with open(json_filename, 'w') as json_file:
    json.dump(formated_arrays, json_file)

json_filename = "/AI/ai_input.json"

# Save the list to a JSON file
with open(json_filename, 'w') as json_file:
    json.dump(formated_arrays, json_file)
