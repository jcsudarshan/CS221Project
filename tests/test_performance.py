##############################################################################
##############################################################################
# Testing suite for algorithms
import baseline
from util_baseline import *

# Current data sets
# Perhaps capture in second and third args
train_path = ""
dev_path = ""

# populate lis of algorithms we'd like to run
current_tests = "baseline"

# Baseline performance
# ---------------------
if "baseline" in current_tests:
	current_alg = "baseline"
	print("Testing " + current_alg + " on " + train_path " and " + dev_path + "...")

	# prelim set-up
	train = util_baseline.readExamples(train_path)
	dev = util_baseline.readExamples(dev_path)
	featureExtractor = baseline.extractWordFeatures
	
	# calculate weights
	weights = baseline.def learnPredictor(train, dev, featureExtractor, eta=0.1, numIters=10)

	# performance analysis
	trainError = evaluatePredictor(trainExamples, lambda(x) : (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))
    devError = evaluatePredictor(devExamples, lambda(x) : (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))