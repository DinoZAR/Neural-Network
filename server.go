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

func LoadTrainingSet(trainingSetFile string) []*TrainTest {
	trainTests := make([]*TrainTest, 0)

	f, err := os.Open(trainingSetFile)
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

func Train(network *Network, trainingSetFile string, learningRate float64, epochs int) {

	trainingSet := LoadTrainingSet(trainingSetFile)

	for e := 0; e < epochs; e++ {

		fmt.Printf("Running epoch %v...\n", e)

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

func ValidateAndReport(network *Network, validationSetFile string) {
	validSet := LoadTrainingSet(validationSetFile)

	for _, test := range validSet {
		network.Process(test.Inputs)
		fmt.Println("--------")
		fmt.Printf("Inputs: %v\n", test.Inputs)
		for i, targetOutput := range test.Outputs {
			fmt.Printf("%v) Expected: %v, Actual: %v\n", i, targetOutput, network.Output(network.OutputNodeID(i)))
		}
	}
}

/**
 * Starts up the server
 */
func main() {

	rand.Seed(20)

	// Create the network
	network := CreateNetwork(3, 3, []int{3, 2})

	for i := 0; i < 10000; i++ {
		network.SetTargetOutputs([]float64{0.2345, 0.5678, 0.98123})
		network.Process([]float64{0.12494, -0.892349, -0.12341})
		network.BackPropagate(0.2)
	}

	network.Process([]float64{0.12494, -0.892349, -0.12341})
	fmt.Printf("Final cost: %v\n", network.Output(network.costNodeID))
	for i := 0; i < network.outputs; i++ {
		fmt.Printf("-> %v) %v\n", i, network.Output(network.OutputNodeID(i)))
	}

	// network.DumpGraph("before_graph.dot")

	// fmt.Println("Training the network...")
	// Train(network, "tests.txt", 0.1, 10)

	// fmt.Println("Validating it...")
	// ValidateAndReport(network, "tests.txt")
	// network.DumpGraph("after_graph.dot")

	fmt.Println("Done!")
}
