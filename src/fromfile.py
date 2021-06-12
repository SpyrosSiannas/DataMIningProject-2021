import csv
import os

# Reads the contents of a csv file and returns the column names
# as well as all the rows
def read_csv(filepath, dataset):
    filepath = "dataset" + os.path.sep + dataset + os.path.sep + filepath + ".csv"
    with open(filepath) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        headers = []
        body = []
        for row in csv_reader:
            if line_count == 0:
                headers = row
            else:
                attrs = []
                for i in range(len(row)):
                    attrs.append(row[i])
                if attrs:
                    body.append(attrs)
            line_count += 1
    return headers, body
