from util_baseline import *
import random

file_path = "../../data/"
file_name = "polarity.dev"
lFile_name = "annotateL_" + file_name 	# labled data
ulFile_name = "annotateUL_" + file_name 	# unlabled data


random.seed(10)
examples = readExamples(file_path + file_name)		# read in examples from data file
sample = random.sample(examples, 50) 			# grab a random sample of 50 items

out = open(file_path + lFile_name, 'w')
for x, y in sample:
	example = str(y) + ' ' + x + '\n'
	out.write(example)
out.close()

out = open(file_path + ulFile_name, 'w')
for x, y in sample:
	example = x + '\n'
	out.write(example)
out.close()