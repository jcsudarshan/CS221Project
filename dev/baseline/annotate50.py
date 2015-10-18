from util_baseline import *
import random

file_path = "../../data/"
file_name = "polarity.dev"
aFile_name = "annotate_" + file_name


random.seed(10)
examples = readExamples(file_path + file_name)		# read in examples from data file
sample = random.sample(examples, 50) 			# grab a random sample of 50 items

out = open(file_path + aFile_name, 'w')
for x, y in sample:
	example = str(y) + ' ' + x + '\n'
	out.write(example)
out.close()