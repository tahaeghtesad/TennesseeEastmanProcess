import csv
import sys
import matplotlib.pyplot as plt

if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise Exception('No file specified')

    path = sys.argv[1]
    weight = .9
    r = []

    with open(path) as input_file:
        with open('../smoothed.csv', 'w') as output_file:
            reader = csv.reader(input_file)
            writer = csv.writer(output_file)

            smoothed = 0.

            i = 0
            for row in reader:
                if i == 0:
                    writer.writerow(row)
                elif i == 1:
                    writer.writerow(row)
                    smoothed = float(row[2])
                else:
                    smoothed = weight * smoothed + (1 - weight) * float(row[2])
                    row[2] = smoothed
                    writer.writerow(row)

                r.append(smoothed)
                i += 1

    plt.plot(r)
    plt.show()


