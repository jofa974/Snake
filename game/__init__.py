def read_training_data():
    training_data = []
    with open("apple_position.in", "r") as f:
        for line in f.readlines():
            training_data.append((int(line.split()[0]), int(line.split()[1])))
    return training_data
