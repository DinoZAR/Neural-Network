package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"math/rand"
	"os"
)

type TrainTest struct {
	Inputs  []float64 `json: inputs`
	Outputs []float64 `json: outputs`
}

func LoadTrainingSet(file string) []*TrainTest {
	trainTests := make([]*TrainTest, 0)

	f, err := os.Open("tests.txt")
	if err != nil {
		fmt.Printf("Error loading tests file: %v", err)
		return nil
	} else {
		scn := bufio.NewScanner(f)
		for scn.Scan() {
			test := &TrainTest{}
			json.Unmarshal([]byte(scn.Text()), test)
			trainTests = append(trainTests, test)
		}
	}
	f.Close()

	return trainTests
}

func Train(network *Network, testFile string, learningRate float64, epochs int) {

	trainingSet := LoadTrainingSet(testFile)

	for e := 0; e < epochs; e++ {

		// Shuffle my training inputs
		for i := range trainingSet {
			j := rand.Intn(i + 1)
			trainingSet[i], trainingSet[j] = trainingSet[j], trainingSet[i]
		}

		// Actually train it
		for _, test := range trainingSet {
			network.SetTargetOutputs(test.Outputs)
			network.Process(test.Inputs)
			network.BackPropagate(learningRate)
		}
	}
}

/**
 * Starts up the server
 */
func main() {

	// Create the network
	network := CreateNetwork(2, 2, []int{10})

	Train(network, "tests.txt", 0.01, 20)

	fmt.Println("Done!")
}
