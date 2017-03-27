package main

import (
	"sync"
)

func (network *Network) BackPropagate(learningRate float64) {

	network.FlushTempData()

	nodeChan := make(chan int, 10000)
	inputChan := make(chan AggregateInputJob, 5)
	var waitGroup sync.WaitGroup

	// Start the back propagate by sending 1 as the partial derivative to the cost node
	inputChan <- AggregateInputJob{
		nodeToAddInput: network.costNodeID,
		input:          1.0,
		inputFromID:    -1,
	}
	waitGroup.Add(1)

	// Process node jobs
	go ProcessBackpropagateNodes(network, learningRate, &waitGroup, nodeChan, inputChan)
	go ProcessBackpropagateNodes(network, learningRate, &waitGroup, nodeChan, inputChan)

	// Aggregate input jobs
	go ProcessAggregateInputJobs(network, &waitGroup, nodeChan, inputChan, false)
	go ProcessAggregateInputJobs(network, &waitGroup, nodeChan, inputChan, false)

	waitGroup.Wait()
	close(nodeChan)
	close(inputChan)
}

func ProcessBackpropagateNodes(network *Network, learningRate float64, waitGroup *sync.WaitGroup, nodeChan chan int, inputChan chan AggregateInputJob) {
	for job := range nodeChan {
		node := network.Node(job)

		// Process the node
		node.lock.Lock()
		var output float64
		sumPartials := SumInputs(node.inputVals)
		if len(node.inputs) > 0 {
			waitGroup.Add(len(node.inputs))
			for _, nodeID := range node.inputs {
				output = node.node.BackPropagate(sumPartials, learningRate, nodeID)
				//fmt.Printf("%v) sumPartials: %v, output: %v\n", job, sumPartials, output)
				inputChan <- AggregateInputJob{
					nodeToAddInput: nodeID,
					input:          output,
					inputFromID:    job,
				}
			}
		} else {
			//fmt.Printf("%v) sumPartials: %v  ---- END\n", job, sumPartials)
			output = node.node.BackPropagate(sumPartials, learningRate, -1)
		}

		node.lock.Unlock()
		waitGroup.Done()
	}
}

func SumInputs(inputs map[int]float64) (result float64) {
	for _, val := range inputs {
		result += val
	}
	return
}
