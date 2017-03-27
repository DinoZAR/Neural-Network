package main

import (
	"math"
	"testing"
)

func notWithinLimits(val, output, threshold float64) bool {
	return math.Abs(val-output) > threshold
}

func TestSigmoid(t *testing.T) {
	var tests = []struct {
		input  float64
		output float64
	}{
		{0.0, 0.5},
		{4.0, 0.98201379},
		{-0.5, 0.37754066},
	}

	for _, test := range tests {
		if notWithinLimits(Sigmoid(test.input), test.output, 0.00000001) {
			t.Fail()
		}
	}
}

func TestTotalCost(t *testing.T) {
	var tests = []struct {
		targetOutputs, actualOutputs map[int]float64
		expectedCost                 float64
	}{
		{
			map[int]float64{
				1: 0.4,
				2: 0.3,
				3: 0.2,
			},
			map[int]float64{
				1: 0.5,
				2: 0.4,
				3: 0.3,
			}, 0.015},
		{
			map[int]float64{
				1: -0.4,
				2: 0.3,
				3: -0.2,
			},
			map[int]float64{
				1: 0.5,
				2: 0.4,
				3: 0.3,
			}, 0.535},
	}

	for _, test := range tests {
		if notWithinLimits(TotalCost(test.targetOutputs, test.actualOutputs), test.expectedCost, 0.00000001) {
			t.Fail()
		}
	}
}

func TestCostInputPartial(t *testing.T) {
	var tests = []struct {
		targetOutput, actualOutput float64
		output                     float64
	}{
		{0.3, -0.5, -0.8},
		{-0.2, 0.0, 0.2},
	}

	for _, test := range tests {
		if notWithinLimits(CostInputPartial(test.targetOutput, test.actualOutput), test.output, 0.000000001) {
			t.Fail()
		}
	}
}

func TestNeuronOutput(t *testing.T) {
	var tests = []struct {
		inputSum, bias float64
		output         float64
	}{
		{-0.4, 0.5, 0.450166003},
		{0.3, 0.4, 0.529964052},
		{-0.2, 0.3, 0.485004498},
	}

	for _, test := range tests {
		if notWithinLimits(NeuronOutput(test.inputSum, test.bias), test.output, 0.000000001) {
			t.Fail()
		}
	}
}

func TestNeuronBiasPartial(t *testing.T) {
	var tests = []struct {
		inputSum, output float64
		expectedOutput   float64
	}{
		{-0.4, 0.5, -0.1},
		{0.3, 0.4, 0.072},
		{-0.2, 0.3, -0.042},
	}

	for _, test := range tests {
		if notWithinLimits(NeuronBiasPartial(test.inputSum, test.output), test.expectedOutput, 0.000000001) {
			t.Fail()
		}
	}
}

func TestNeuronInputPartial(t *testing.T) {
	var tests = []struct {
		bias, output   float64
		expectedOutput float64
	}{
		{-0.4, 0.5, -0.1},
		{0.3, 0.4, 0.072},
		{-0.2, 0.3, -0.042},
	}

	for _, test := range tests {
		if notWithinLimits(NeuronBiasPartial(test.bias, test.output), test.expectedOutput, 0.000000001) {
			t.Fail()
		}
	}
}
