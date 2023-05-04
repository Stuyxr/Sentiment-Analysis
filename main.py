import numpy as np

def get_prediction(index, file="predicted_results.csv"):
    predicted_results = np.loadtxt(file, delimiter=",")
    return predicted_results[index]

input_index = 0  # Choose the index of the input you want to check
prediction = get_prediction(input_index)
print(f"Predicted sentiment for input {input_index}: {prediction}")
