##############################################################################
##############################################################################
# Testing suite for algorithms
#import algorithms.baseline

import baseline
import util_baseline
import heuristics as heur
import random
from os import listdir
from os.path import join


# Current data sets
# Perhaps capture in second and third args
# current dir is CS221Project/tests/
dataDir = "../../data"
testDir = "../../test"
train_path = "../../data/polarity.train"
dev_path = "../../data/annotateL_polarity.dev"
rawTwitterTrainingPath = "../../data/training.1600000.processed.noemoticon.csv"
modifiedTwitterTrainingPath = "../../data/training.modified.1600000.processed.noemoticon.csv"
rawTwitterTestPath = "../../data/testdata.manual.2009.06.14.csv"
modifiedTwitterTestPath = "../../data/testdata.modified.manual.2009.06.14.csv"
oracleAnnotationsDir = "../../data/initial_baseline_oracle_tests"
oracleCorrectLabels = "../../data/annotateL_polarity.dev"
rawUmichPath = "../../data/UM_ContrivedSentiment"
modifiedUmichPath = "../../data/UM_ContrivedSentiment_modified"
trainUmichPath = "../../data/UM_ContrivedSentiment_train"
devUmichPath = "../../data/UM_ContrivedSentiment_dev"

# populate lis of algorithms we'd like to run
current_tests = "baseline"

# Baseline performance
# ---------------------
def runBaseline():
	current_alg = "baseline"
	print("Testing " + current_alg + " on " + train_path + " and " + dev_path + "...")

	# prelim set-up
	train = util_baseline.readExamples(train_path)
	dev = util_baseline.readExamples(dev_path)
	featureExtractor = baseline.extractWordFeatures

	# calculate weights
	weights = baseline.learnPredictor(train, dev, featureExtractor, eta=0.1, numIters=10)

	# performance analysis
	trainError = util_baseline.evaluatePredictor(train, lambda(x) : \
		(1 if util_baseline.dotProduct(featureExtractor(x), weights) >= 0 else -1))
	devError = util_baseline.evaluatePredictor(dev, lambda(x) : \
    	(1 if util_baseline.dotProduct(featureExtractor(x), weights) >= 0 else -1))

	print "Error percentage on train examples: " + str(trainError)
	print "Error percentage on dev examples: " + str(devError)

	util_baseline.outputErrorAnalysis(train, featureExtractor, weights, testDir, 'train-error-analysis')
	util_baseline.outputErrorAnalysis(dev, featureExtractor, weights, testDir, 'dev-error-analysis')

# Window Heuristic performance
# ---------------------
def runWindowHeuristic():
	current_alg = "windowHeuristic"
	print("Testing " + current_alg + " on " + train_path + " and " + dev_path + "...")

	# prelim set-up
	train = util_baseline.readExamples(train_path)
	dev = util_baseline.readExamples(dev_path)
	featureExtractor = baseline.extractWordFeatures

	# calculate weights
	weights = baseline.learnPredictor(train, dev, featureExtractor, eta=0.1, numIters=10)

	# performance analysis
	trainError = util_baseline.evaluatePredictor(train, lambda(x) : \
		(1 if heur.windowPredictor(x, featureExtractor(x), weights) >= 0 else -1))
	devError = util_baseline.evaluatePredictor(dev, lambda(x) : \
		(1 if heur.windowPredictor(x, featureExtractor(x), weights) >= 0 else -1))

	print "Error percentage on train examples: " + str(trainError)
	print "Error percentage on dev examples: " + str(devError)

	util_baseline.outputErrorAnalysis(train, featureExtractor, weights, testDir, 'train-error-analysis')
	util_baseline.outputErrorAnalysis(dev, featureExtractor, weights, testDir, 'dev-error-analysis')


if "baseline" in current_tests:
	pass #runBaseline()

def convertTwitterSetToNormalForm(cleanData, rawPath, modifiedPath):
	def includeToken(word):
		if word[0] == '@' or word[0] == '#':
			return False
		if word[0:4] == 'http':
			return False
		return True
	training = open(rawPath)
	modified = open(modifiedPath, 'w')

	for line in training:
		polarity, id, date, query, user, sentence = line.split(',', 5)

		if polarity == '"0"':
			newPolarity = -1
		elif polarity == '"4"':
			newPolarity = 1
		elif polarity == '"2"':
			newPolarity = 0

		modifiedSentence = sentence[1:len(sentence) - 2]
		if cleanData:
			words = modifiedSentence.split()
			modifiedSentence = ' '.join(word for word in words if includeToken(word))
		print >>modified, "%d %s" % (newPolarity, modifiedSentence)

	training.close()
	modified.close()
#convertTwitterSetToNormalForm(True, rawTwitterTestPath, modifiedTwitterTestPath)


def convertUMichToNormalForm(rawPath, modifiedPath):
	"""
	Converts University of Michigan data from in class kaggle:
	https://inclass.kaggle.com/c/si650winter11/data
	"""
	training = open(rawPath)
	modified = open(modifiedPath, 'w')

	for line in training:
		polarity, sentence = line.split('\t')
		sentence = sentence.rstrip('\n')

		if int(polarity) == 0:
			newPolarity = -1
		else:
			newPolarity = 1

		print >>modified, "%d %s" % (newPolarity, sentence)

	training.close()
	modified.close()

# convertUMichToNormalForm(rawUmichPath, modifiedUmichPath)


def createUmichTrainAndDev(originalDataPath, trainPath, devPath):
	"""
	Shuffles and seperates UMich into training set (n=6,073)
	and dev set (n=1013).
	"""
	originalData = open(originalDataPath)
	training = open(trainPath, 'w')
	dev = open(devPath, 'w')

	# Shuffle lines
	data = [(random.random(), line) for line in originalData]
	data.sort()

	for line_num in range(len(data)):
		# select ~ 1,000 dev examples (total is 7086)
		if line_num % 7 == 0:
			print >>dev, data[line_num][1].rstrip('\n')
		else:
			print >>training, data[line_num][1].rstrip('\n')

	originalData.close()
	training.close()
	dev.close()
# createUmichTrainAndDev(modifiedUmichPath, trainUmichPath, devUmichPath)


def determineOracleAgreement(oracleAnnotationsDir):
	fileHandles = [open(join(oracleAnnotationsDir, f)) for f in listdir(oracleAnnotationsDir) if f.find('polarity') >= 0]
	answerLabels = open(oracleCorrectLabels)
	correctOut = open(join(testDir, 'correct-oracle-judgements.csv'), 'w')
	wrongOut = open(join(testDir, 'wrong-oracle-judgements.csv'), 'w')

	for example in zip(fileHandles[0], fileHandles[1], fileHandles[2]):
		groupConcensus = sum(int(decision.split()[0]) for decision in example)
		answerLabel = int(answerLabels.readline().split()[0])
		out = correctOut if groupConcensus * answerLabel >= 0 else wrongOut
		print >>out, "Answer: %d, Group concensus: %d, Original sentence: '%s'" % \
							(answerLabel, groupConcensus, ' '.join(example[0].split()[1:-1]))

	for handle in fileHandles:
		handle.close()



#determineOracleAgreement(oracleAnnotationsDir)

runWindowHeuristic()
#runBaseline()