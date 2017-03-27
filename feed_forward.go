package main

import (
	"sync"
)

type AggregateInputJob struct {
	nodeToAddInput int
	input          float64
	inputFromID    int
}

func (network *Network) SetTargetOutputs(targets []float64) {
	node := network.Node(network.costNodeID).node.(*CostNode)

	// Convert the array into a map pointing to the specific output nodes its targeting
	targetMap := make(map[int]float64)
	for i, t := range targets {
		targetMap[network.outputStart+i] = t
	}

	// Save that into the node
	node.targetOutputs = targetMap
}

func (network *Network) Process(inputs []float64) {

	network.FlushTempData()

	nodeChan := make(chan int, 10000)
	inputChan := make(chan AggregateInputJob, len(inputs))
	var waitGroup sync.WaitGroup

	// Add inputs as initial feed jobs
	for i, val := range inputs {
		inputChan <- AggregateInputJob{
			nodeToAddInput: i,
			input:          val,
			inputFromID:    -1,
		}
	}
	waitGroup.Add(network.inputs)

	// Process node jobs
	go ProcessNodeJobs(network, &waitGroup, nodeChan, inputChan)
	go ProcessNodeJobs(network, &waitGroup, nodeChan, inputChan)

	// Aggregate input jobs
	go ProcessAggregateInputJobs(network, &waitGroup, nodeChan, inputChan, true)
	go ProcessAggregateInputJobs(network, &waitGroup, nodeChan, inputChan, true)

	// Wait until all nodes are processed
	waitGroup.Wait()
	close(nodeChan)
	close(inputChan)
}

func ProcessNodeJobs(network *Network, waitGroup *sync.WaitGroup, jobChan chan int, inputChan chan AggregateInputJob) {
	for job := range jobChan {
		node := network.Node(job)

		// Process the node
		node.lock.Lock()
		output := node.node.FeedForward(node.inputVals)
		outputNodes := node.outputs
		node.lock.Unlock()

		// Save the output
		network.SaveOutput(job, output)

		// Send out its output
		waitGroup.Add(len(outputNodes)) // Add the aggregates inputs
		for _, outNode := range outputNodes {
			inputChan <- AggregateInputJob{
				nodeToAddInput: outNode,
				input:          output,
				inputFromID:    job,
			}
		}

		waitGroup.Done() // Finish the node
	}
}

func ProcessAggregateInputJobs(network *Network, waitGroup *sync.WaitGroup, jobChan chan int, inputChan chan AggregateInputJob, inputToOutput bool) {
	for inputJob := range inputChan {
		node := network.Node(inputJob.nodeToAddInput)
		node.lock.Lock()

		node.inputVals[inputJob.inputFromID] = inputJob.input

		// Depending on which direction we're going, set what length we're comparing against
		var compare int
		if inputToOutput {
			compare = len(node.inputs)
		} else {
			compare = len(node.outputs)
		}

		if len(node.inputVals) >= compare {
			waitGroup.Add(1)
			jobChan <- node.id
		}

		node.lock.Unlock()
		waitGroup.Done()
	}
}
