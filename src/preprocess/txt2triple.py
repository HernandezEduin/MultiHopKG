import csv
import argparse
import sys

parser = argparse.ArgumentParser(description="Convert a CSV file to a tab-separated values (TSV) file.")
parser.add_argument("input_file", type=str, help="Path to the input CSV file.")
parser.add_argument("output_file", type=str, help="Path to the output TSV file.")
args = parser.parse_args()

with open(args.input_file, "r") as csv_file:
    csv_reader = csv.reader(csv_file)
    with open(args.output_file, "w") as tsv_file:
        for row in csv_reader:
            # Assuming the first column is the subject, the second is the predicate, and the third is the object
            # row contains \t and \n
            # Remove any unwanted characters from the subject, predicate, and object
            row = row[0].strip().split()
            subject = row[0]
            predicate = row[1]
            obj = row[2]
            # Write to TSV file
            tsv_file.write(f"{subject}\t{obj}\t{predicate}\n")
