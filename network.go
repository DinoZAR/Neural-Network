package main

import (
	"bytes"
	"fmt"
	"math/rand"
	"sync"
)

type NetworkNode struct {
	id        int
	inputs    []int
	outputs   []int
	inputVals map[int]float64
	node      Feedable
	lock      sync.Mutex
}

type Network struct {
	nodes          map[int]*NetworkNode
	nodesLock      sync.RWMutex
	inputs         int
	outputs        int
	outputStart    int
	costNodeID     int
	outputVals     map[int]float64
	outputValsLock sync.RWMutex
}

func (n *Network) FlushTempData() {
	for _, node := range n.nodes {
		node.inputVals = make(map[int]float64)
	}
}

func (n *Network) Node(nodeID int) *NetworkNode {
	n.nodesLock.RLock()
	node := n.nodes[nodeID]
	n.nodesLock.RUnlock()
	return node
}

func (n *Network) SaveOutput(nodeID int, output float64) {
	n.outputValsLock.Lock()
	n.outputVals[nodeID] = output
	n.outputValsLock.Unlock()
}

func (n *Network) Output(nodeID int) float64 {
	n.outputValsLock.RLock()
	output := n.outputVals[nodeID]
	n.outputValsLock.RUnlock()
	return output
}

func (n *Network) OutputNodeID(outputIdx int) int {
	return n.outputStart + outputIdx
}

func CreateNetwork(inputs int, outputs int, hiddenLayers []int) *Network {

	nodes := make(map[int]*NetworkNode)
	idGen := -1

	// Input nodes
	for i := 0; i < inputs; i++ {
		net := WrapNode(&idGen, &NeuronNode{
			bias:        rand.Float64()*2 - 1,
			inputSum:    0,
			savedOutput: 0,
		})
		nodes[idGen] = net
	}

	// Hidden layers and the connections between all of them
	lastStart := 0
	lastEnd := idGen
	for _, numHidden := range hiddenLayers {

		// Create the hidden neurons
		hiddenStart := idGen + 1
		for i := 0; i < numHidden; i++ {
			net := WrapNode(&idGen, &NeuronNode{
				bias:        rand.Float64()*2 - 1,
				inputSum:    0,
				savedOutput: 0,
			})
			nodes[idGen] = net
		}
		hiddenEnd := idGen

		// Create the weighted connections between previous layer and this one
		for i := 0; i < (lastEnd - lastStart + 1); i++ {
			for j := 0; j < numHidden; j++ {
				net := WrapNode(&idGen, &WeightedNode{rand.Float64()*2 - 1})
				nodes[idGen] = net
				Connect(nodes[lastStart+i], nodes[idGen])   // to previous layer
				Connect(nodes[idGen], nodes[hiddenStart+j]) // to this layer
			}
		}

		// Set up variables for next loop
		lastStart = hiddenStart
		lastEnd = hiddenEnd
	}

	// Output neurons
	outputStart := idGen + 1
	for i := 0; i < outputs; i++ {
		net := WrapNode(&idGen, &NeuronNode{
			bias:        rand.Float64()*2 - 1,
			inputSum:    0,
			savedOutput: 0,
		})
		nodes[idGen] = net
	}

	// And their connections to last layer in hidden layer
	for i := 0; i < (lastEnd - lastStart + 1); i++ {
		for j := 0; j < outputs; j++ {
			net := WrapNode(&idGen, &WeightedNode{rand.Float64()*2 - 1})
			nodes[idGen] = net
			Connect(nodes[lastStart+i], nodes[idGen])
			Connect(nodes[idGen], nodes[outputStart+j])
		}
	}

	// Finally, connect the cost node
	net := WrapNode(&idGen, &CostNode{
		targetOutputs: make(map[int]float64),
		actualOutputs: make(map[int]float64),
	})
	nodes[idGen] = net
	for i := 0; i < outputs; i++ {
		Connect(nodes[outputStart+i], net)
	}

	// Return it all
	return &Network{
		inputs:      inputs,
		outputs:     outputs,
		nodes:       nodes,
		outputStart: outputStart,
		costNodeID:  idGen,
		outputVals:  make(map[int]float64),
	}
}

func WrapNode(idGen *int, node Feedable) *NetworkNode {
	*idGen++
	net := NetworkNode{
		node:      node,
		inputs:    make([]int, 0),
		outputs:   make([]int, 0),
		inputVals: make(map[int]float64),
		id:        *idGen,
	}
	return &net
}

func Connect(leftNode, rightNode *NetworkNode) {
	leftNode.outputs = append(leftNode.outputs, rightNode.id)
	rightNode.inputs = append(rightNode.inputs, leftNode.id)
}

func (network *Network) String() string {
	var buff bytes.Buffer

	// Top-level stuff
	buff.WriteString(fmt.Sprintf("{inputs: %v, outputs: %v, outputStart: %v, costNodeID: %v}\n", network.inputs, network.outputs, network.outputStart, network.costNodeID))

	// Now the actual nodes
	for id, node := range network.nodes {
		buff.WriteString(fmt.Sprintf("-> id: %v, node: %v\n", id, node))
	}

	return buff.String()
}
