import matplotlib.pyplot as plt
import numpy as np
import random as random

data_path = "Labs/Laboration-2/datapoints.txt"
test_path = "Labs/Laboration-2/testpoints.txt"

def load_data_points(path): # Läs och förbehandlar datapunkter (plockar bort ex. mellanslag) och sätter in de i values x, y och label

    points = []

    with open(path, "r") as file:
        next(file)
        for line in file:
            try:
                line = line.strip().split(",")
                x, y, label = line

                x = float(x)
                y = float(y)
                label = int(label)

                if x < 0 or y < 0:
                    raise ValueError(f"Negative number found on line {line}")
            
                points.append((float(x), float(y), int(label)))                
            except ValueError as error:
                print(f"Error on line {line}: {error}")

    return points

def load_test_points(path): # Läs och förbehandlar datapunkter (plockar bort ex. mellanslag) och sätter in de i values x, y

    points = []

    with open(path, "r") as file:
        next(file)
        for line in file:
            try:    
                line = line.strip()
                line = line.split("(")[1].strip(")").split(",")
                x, y = line

                x = float(x)
                y = float(y)

                if x < 0 or y < 0:
                    raise ValueError(f"Negative number found on line {line}")
            
                points.append((float(x), float(y)))
            except ValueError as error:
                print(f"Error on line {line}: {error}")

    return points

def euclidean_distance(p1, p2): #Beräknar euklidiskt avstånd, används sedan för klassificering 
    euc_dis = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    return euc_dis

def classification_1NN(test_point, training_data): #Klassificering för allra närmsta granne
    distances = []
    for (x, y, label) in training_data:
        distance = euclidean_distance(test_point, (x, y))
        distances.append((distance, label))

    min_distance, closest_label = min(distances, key=lambda d: d[0])
    return closest_label

def classification_kNN(test_point, training_data, k=10): # Klassificering av K-närmsta granne, i detta fall 10 närmsta
    distances = []
    for x, y, label in training_data:
        distance = euclidean_distance(test_point, (x, y))
        distances.append((distance, label))

    sorted_distances = sorted(distances, key=lambda d: d[0])
    k_nearest = sorted_distances[:k]
    neighbor_labels = [label for _, label in k_nearest]

    most_common = max(set(neighbor_labels), key=neighbor_labels.count)
    return most_common

def plot_data_1NN(training_data, test_results): # Plottar / visualiserar samlad och hanterad data
    plt.figure("1-NN Classification")
    for x, y, label in training_data:
            plt.scatter(x, y, color="blue" if label == 0 else "orange")
    for x, y, label in test_results:
            plt.scatter(x, y, color="red" if label == 0 else "green", marker="x", s=100)
    
    plt.title("Pikachu vs Pichu with 1-NN")
    plt.xlabel("Width (cm)")
    plt.ylabel("Height (cm)")
    plt.grid(True)
    plt.text(16.5, 40, "Pikachu = Orange", fontsize=10)
    plt.text(16.5, 39, "Pichu = Blue", fontsize=10)
    plt.text(16.5, 38, "Test point identified Pichu = Red", fontsize=8)
    plt.text(16.5, 37.5, "Test point identified Pikachu = Green", fontsize=8)

def plot_data_kNN(training_data, test_results, k): # Plottar / visualiserar samlad och hanterad data
    plt.figure(f"{k}-NN Classification")
    for x, y, label in training_data:
            plt.scatter(x, y, color="blue" if label == 0 else "orange")
    for x, y, label in test_results:
            plt.scatter(x, y, color="red" if label == 0 else "green", marker="x", s=100)

    plt.title(f"Pikachu vs Pichu with {k}-NN")
    plt.xlabel("Width (cm)")
    plt.ylabel("Height (cm)")
    plt.grid(True)
    plt.text(16.5, 40, "Pikachu = Orange", fontsize=10)
    plt.text(16.5, 39, "Pichu = Blue", fontsize=10)
    plt.text(16.5, 38, "Test point identified Pichu = Red", fontsize=8)
    plt.text(16.5, 37.5, "Test point identified Pikachu = Green", fontsize=8)

def classify_user_input(training_data, k = 10): # Kod för att klassificera användarens egna inputs
    print("Classification of user input")
    try:
        width = float(input("Enter width of the pokemon (cm): "))
        height = float(input("Enter height of the pokemon (cm): "))

        if width < 0 or height < 0:
            raise ValueError("Negative number found, please enter a positive nunber.")
        
        label = classification_kNN((width, height), training_data, k)
        label_name = "Pikachu" if label == 1 else "Pichu"
        print(f"Test point with (width, height): {width}, {height} identified as: {label_name}")

    except ValueError as error:
        print(f"Error input: {error}")

def split_data(data, train_size = 50): # Splitta data för att sedan beräkna exakthet (accuracy)
        pikachu = [point for point in data if point[2] == 1]
        pichu = [point for point in data if point[2] == 0]

        random.shuffle(pikachu)
        random.shuffle(pichu)

        training_data = pikachu[:train_size] + pichu[:train_size]
        test_data = pikachu[train_size:train_size + 25] + pichu[train_size:train_size + 25]

        random.shuffle(training_data)
        random.shuffle(test_data)

        return training_data, test_data
    
def calc_accuracy(predict_labels, real_labels): # Beräkna accuracy
    TP = FP = TN = FN = 0

    for predict, real in zip(predict_labels, real_labels):
        if predict == 1 and real == 1:
            TP += 1
        elif predict == 1 and real == 0:
            FP += 1
        elif predict == 0 and real == 0:
            TN += 1
        elif predict == 0 and real == 1:
            FN += 1

    total = TP + TN + FP + FN
    accuracy = (TP + TN) / total
    return accuracy 

def run_accuracy(data, k = 10, iteration = 10): #Kör igenom med all hanterad accuracy data
    accuracies = []

    for _ in range(iteration):
        training_data, test_data = split_data(data)

        predict_labels = []
        real_labels = []

        for x, y, real_label in test_data:
            predict_label = classification_kNN((x, y), training_data, k)
            predict_labels.append(predict_label)
            real_labels.append(real_label)
        accuracy = calc_accuracy(predict_labels, real_labels)
        accuracies.append(accuracy)
    return accuracies

def plot_accuracy(accuracies): #Plottar och visualiserar exakhet efter att ha kört 10 gånger
    plt.figure("Accuracy after 10 runs")
    plt.plot(range(1, len(accuracies)+1), accuracies, marker='o')
    plt.xlabel("Run")
    plt.ylabel("Accuracy")
    plt.title("Accuracy of k-NN after 10 runs")
    plt.grid(True)

def main(): # Main-program (of course)
    print("Classification of test points:")
    test_results = []

    training_data = load_data_points(data_path)
    test_points = load_test_points(test_path)
    k = 10
    test_results_knn = []

    for point in test_points:
        label = classification_1NN(point, training_data)
        label_name = "Pikachu" if label == 1 else "Pichu"
        print(f"1-NN: Sample with (width, height): {point} classified as {label_name}")
        test_results.append((point[0], point[1], label))

    for point in test_points:
        label = classification_kNN(point, training_data, k)
        label_name = "Pikachu" if label == 1 else "Pichu"
        print(f"{k}-NN: Sample with (width, height): {point} classified as {label_name}")
        test_results_knn.append((point[0], point[1], label))
    
    plot_data_1NN(training_data, test_results)
    plot_data_kNN(training_data, test_results_knn, k)
    
    print("k-NN accuracy experiment")
    all_data = load_data_points(data_path)
    accuracies = run_accuracy(all_data, k = 10, iteration = 10)
    total_accuracy = sum(accuracies) / len(accuracies)
    print(f"Average accuracy after 10 runs: {total_accuracy:.2%}")
    plot_accuracy(accuracies)

    print("Remove plot windows to start classification of user input")

    plt.show()

    classify_user_input(training_data, k)

    # Öppnar först tre fönster första med testpunkterna efter man hittat allra närmsta granne,
    # Sedan andra fönster som identifierar testpunkterna efter 10 närmasta grannar,
    # Och sedan sista fönstret som beräknar "accuracy" av denna algoritm.
    # Efter detta kommer det upp i terminalen för egen inmatning av bredd och höjd för att identifiera Pikachu eller Pichu

if __name__ == "__main__":
    main()