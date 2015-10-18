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

# populate lis of algorithms we'd like to run
current_tests = "baseline"

# Baseline performance
# ---------------------
if "baseline" in current_tests:
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