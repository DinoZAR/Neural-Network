package main

import (
	"math"
)

/**
 * General-purpose functions
 */
func Sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

func Learn(currentVal, learningRate, partial float64) float64 {
	return currentVal - (learningRate * partial)
}

/**
 * Interface for every node to implement going backwards and forwards in the network
 */
type Feedable interface {
	FeedForward(inputs map[int]float64) (output float64)
	BackPropagate(partialSum float64, learningRate float64, inputNodeID int) (outputPartial float64)
}

/*
 * Cost node functions
 */
func TotalCost(targetOutputs map[int]float64, actualOutputs map[int]float64) (result float64) {
	for nodeID, target := range targetOutputs {
		result += 0.5 * math.Pow(target-actualOutputs[nodeID], 2)
	}
	return
}

func CostInputPartial(targetOutput, actualOutput float64) float64 {
	return -(targetOutput - actualOutput)
}

/*
 * Cost node and its implementing methods to Feedable
 */
type CostNode struct {
	targetOutputs map[int]float64
	actualOutputs map[int]float64
}

func (n *CostNode) FeedForward(inputs map[int]float64) (output float64) {
	n.actualOutputs = inputs
	return TotalCost(n.targetOutputs, n.actualOutputs)
}

func (n *CostNode) BackPropagate(partialSum float64, learningRate float64, inputNodeID int) (outputPartial float64) {
	return CostInputPartial(n.targetOutputs[inputNodeID], n.actualOutputs[inputNodeID])
}

/*
 * Weighted node and its implementing methods to Feedable
 */
type WeightedNode struct {
	weight float64
}

func (n *WeightedNode) FeedForward(inputs map[int]float64) (output float64) {
	return InputSum(inputs) * n.weight
}

func (n *WeightedNode) BackPropagate(partialSum float64, learningRate float64, inputNodeID int) (outputPartial float64) {
	out := n.weight * partialSum
	n.weight = Learn(n.weight, learningRate, out)
	return out
}

/*
 * Neuron node functions
 */
func NeuronOutput(inputSum, bias float64) float64 {
	return Sigmoid(inputSum * bias)
}

func InputSum(inputs map[int]float64) (result float64) {
	for _, i := range inputs {
		result += i
	}
	return
}

func NeuronBiasPartial(inputSum, output float64) float64 {
	return inputSum * output * (1 - output)
}

func NeuronInputPartial(bias, output float64) float64 {
	return bias * output * (1 - output)
}

/*
 * Neuron node and its implementing methods to Feedable
 */
type NeuronNode struct {
	bias, inputSum, savedOutput float64
}

func (n *NeuronNode) FeedForward(inputs map[int]float64) (output float64) {
	n.inputSum = InputSum(inputs)
	n.savedOutput = NeuronOutput(n.inputSum, n.bias)
	return n.savedOutput
}

func (n *NeuronNode) BackPropagate(partialSum float64, learningRate float64, inputNodeID int) (outputPartial float64) {

	// Calculate the output partial with what I have now
	inputPartial := NeuronInputPartial(n.bias, n.savedOutput) * partialSum
	biasPartial := NeuronBiasPartial(n.inputSum, n.savedOutput) * partialSum

	// Update the bias now
	n.bias = Learn(n.bias, learningRate, biasPartial)

	return inputPartial
}
