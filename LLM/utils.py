

def result_to_file(data, filename):
    with open(filename, "w", encoding="utf-8") as file:
        file.write(data)
