import csv
import numpy as np

def extractData(filename):
    """ Reads data from csv file and returns it in array form.

    Parameters
    ----------
    filename : str
        File path of data file to read

    Returns
    -------
    data : arr

    """
    lst=[]
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            lst.append(row)
    data=np.array(lst, dtype=float)
    return data

def writeResult(filename, data):
    """ Writes data predicted by trained algorithm into a csv file.

    Parameters
    ----------
    filename : str
        File path of data file to read

    data : arr
        Array of data to write in csv file

    """
    with open(filename, 'w') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',')
        for row in data:
            spamwriter.writerow(row)
