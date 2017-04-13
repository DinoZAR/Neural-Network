# Create a bunch of test cases for simple functions to feed into a neural network.
import random
import json
import math

NUM_TESTS = 1000
TEST_FILE = 'tests.txt'

def save_test(inputs, outputs, f):
	
	# Create the json
	stuff = {
		'inputs': inputs,
		'outputs': outputs
	}
	f.write(json.dumps(stuff) + '\n')

if __name__ == '__main__':

	with open(TEST_FILE, 'w') as f:
		for i in range(NUM_TESTS):

			x = (i / NUM_TESTS) * 2 * math.pi
			y = math.sin(x)

			save_test([x], [y], f)

	print('Done!')