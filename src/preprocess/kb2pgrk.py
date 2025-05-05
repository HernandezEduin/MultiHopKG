import csv
import argparse
import sys
# to calculcate probability of each entity
import collections

'''
counts probability of each entity
'''

parser = argparse.ArgumentParser(description="Convert a CSV file to a tab-separated values (TSV) file.")
parser.add_argument("input_file", type=str, help="Path to the input CSV file.")
parser.add_argument("output_file", type=str, help="Path to the output TSV file.")
args = parser.parse_args()

entities = []
with open(args.input_file, "r") as csv_file:
    csv_reader = csv.reader(csv_file)
    for row in csv_reader:
        # Assuming the first column is the subject, the second is the predicate, and the third is the object
        # row contains \t and \n
        # Remove any unwanted characters from the subject, predicate, and object
        row = row[0].strip().split()
        subject = row[0]
        obj = row[1]
        predicate = row[2]
        entities.append(subject)
        entities.append(obj)

# Count the frequency of each entity
entity_counts = collections.Counter(entities)
# Calculate the total number of entities
total_entities = len(entities)
# Calculate the probability of each entity
entity_probabilities = {entity: count / total_entities for entity, count in entity_counts.items()}

# format is like this
# entity1                       :0.00993494942493237
# entity2                       :0.009934949421348163
# entity3                       :0.009852609895660556
# entity4                       :0.009852606777536856

# sort the dictionary by value from high to low
entity_probabilities = dict(sorted(entity_probabilities.items(), key=lambda item: item[1], reverse=True))

with open(args.output_file, "w") as tsv_file:
    for entity, probability in entity_probabilities.items():
        # Write to TSV file
        tsv_file.write(f"{entity}\t:{probability}\n")
# to calculate probability of each entity
