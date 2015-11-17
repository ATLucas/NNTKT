package tools;

import containers.Matrix;
import containers.Example;
import containers.Dataset;
import network.NeuralNetwork;
import readers.MnistReader;

/**
 * Created by Andrew on 11/11/2015.
 */
public class MnistTrainer {

	public static void main(String args[]) {
		Logger.init("log/mnist");

		Dataset trainData = new Dataset();
		MnistReader.Read(trainData, "res/mnist/train-images.idx3-ubyte", "res/mnist/train-labels.idx1-ubyte");

		Dataset testData = new Dataset();
		MnistReader.Read(testData, "res/mnist/t10k-images.idx3-ubyte", "res/mnist/t10k-labels.idx1-ubyte");

		TrainConfig trainConfig= new TrainConfig(args[0], 60000);
		Logger.log(""+trainConfig);

		NeuralNetwork network = new NeuralNetwork(trainConfig.getTopology());

		Train(trainConfig, network, trainData, testData);

		Logger.close();
	}

	public static void Train(TrainConfig config, NeuralNetwork network, Dataset trainData, Dataset testData) {
		int epoch = 0;
		for (int i = 0; i < config.numEpochs * config.itersPerEpoch; i++) {
			if(i > 0 && i % config.itersBetweenWritingModel == 0) {
				Common.write("models/mnist/" + Logger.timeString + "." + i + ".dnn", network.toString());
			}

			if (i % config.itersPerEpoch == 0) {
				Logger.log("\nEpoch " + (++epoch) + " of " + config.numEpochs);
			}
			Logger.log("\nIteration " + (i + 1) + " of " + config.numEpochs * config.itersPerEpoch + " at " + Logger.time());

			float cost = 0;
			for (int m=0; m<config.batchesPerIter; m++) {
				cost += network.trainMinibatch(config, trainData.getMinibatch(config.minibatchSize));
			}
			Logger.log("Average cost per example: " + (cost / config.batchesPerIter) + "\n");

			Logger.log("Validating at " + Logger.time() + "...");
			//Logger.log(" training set...");
			//Decode(network, trainData);
			Logger.log(" validation set...");
			Decode(network, testData);

			Logger.flush();
//			if(config.shouldWriteModel(i)) {
//				network.writeToFile("models/"+logger.timeString+"_"+i+".nnet");
//			}
		}
	}

	public static void Decode(NeuralNetwork network, Dataset dataset) {
		int numCorrect = 0;
		for(int i=0; i<dataset.size(); i++) {
			Example example = dataset.getExample(i);
			Matrix input = new Matrix(1, network.inputDim());
			input.copyRow(example.input);
			Matrix output = network.forward(input);
			Matrix target = new Matrix(1, network.outputDim());
			target.copyRow(example.target);
			if(output.getIndexOfMaxElement() == target.getIndexOfMaxElement()) numCorrect++;
		}
		Logger.log("Accuracy: "+((float)numCorrect*100f/(float)dataset.size()));
	}

}
