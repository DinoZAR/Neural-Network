# Create a bunch of test cases for simple functions to feed into a neural network.
import random
import json

NUM_TESTS = 1000
TEST_FILE = 'tests.csv'


def function(inputs):
	"""
	Given its [inputs], return corresponding [outputs]
	"""
	
	# This will be an AND boolean function
	# 1, 0 for true
	# 0, 1 for false
	val1, val2 = inputs

	if val1 and val2:
		return 0.9, 0.1
	else:
		return 0.1, 0.9

def save_test(inputs, outputs, f):
	
	# Create the json
	stuff = {
		'inputs': inputs,
		'outputs': outputs
	}
	f.write(json.dumps(stuff) + ';')

if __name__ == '__main__':

	with open(TEST_FILE, 'w') as f:
		for i in range(NUM_TESTS):
			val1 = random.choice([True, False])
			val2 = random.choice([True, False])
			inputs = [0.9 if val1 else 0.1, 0.9 if val2 else 0.1]

			outputs = function(inputs)

			save_test(inputs, outputs, f)

	print('Done!')