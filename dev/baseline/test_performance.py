##############################################################################
##############################################################################
# Testing suite for algorithms
#import algorithms.baseline

import baseline
import util_baseline

# Current data sets
# Perhaps capture in second and third args
# current dir is CS221Project/tests/
train_path = "../../data/polarity.train"
dev_path = "../../data/annotate_polarity.dev"
rawTwitterTrainingPath = "../../data/training.1600000.processed.noemoticon.csv"
modifiedTwitterTrainingPath = "../../data/training.modified.1600000.processed.noemoticon.csv"
rawTwitterTestPath = "../../data/testdata.manual.2009.06.14.csv"
modifiedTwitterTestPath = "../../data/testdata.modified.manual.2009.06.14.csv"

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

	util_baseline.outputErrorAnalysis(train, featureExtractor, weights, 'train-error-analysis')
	util_baseline.outputErrorAnalysis(dev, featureExtractor, weights, 'dev-error-analysis')


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