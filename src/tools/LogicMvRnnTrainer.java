package tools;

import containers.Forest;
import containers.Matrix;
import containers.TreeExample;
import network.NeuralNetwork;
import nlp.Vocab;
import org.json.JSONObject;
import readers.ForestReader;

/**
 * Created by Andrew on 11/15/2015.
 */
public class LogicMvRnnTrainer {

	public static void main(String args[]) {
		Logger.init("log/logic");

		TrainConfig trainVectorConfig= new TrainConfig(args[0], 6);
		Logger.log("Train Vector Config:\n" + trainVectorConfig);

		Vocab vocab = null;
		try {
			vocab = new Vocab(new JSONObject(Common.readFileIntoString(args[2])));
		} catch(Exception e){e.printStackTrace();}

		Forest forest = new Forest();
		ForestReader.Read(forest, vocab, args[3]);

		Train(trainVectorConfig, forest);

		Logger.close();
	}

	public static void Train(TrainConfig config, Forest forest) {
		NeuralNetwork network = new NeuralNetwork(config.getTopology());
		Matrix weights = new Matrix(2,1);//todo shouldn't be hard-coded

		int epoch = 0;
		for (int i = 0; i < config.numEpochs * config.itersPerEpoch; i++) {
			if (i > 0 && i % config.itersBetweenWritingModel == 0) {
				Common.write("models/logic/" + Logger.timeString + "." + i + ".v.dnn", network.toString());
			}

			if (i % config.itersPerEpoch == 0) {
				Logger.log("\nEpoch " + (++epoch) + " of " + config.numEpochs);
			}
			Logger.log("\nIteration " + (i + 1) + " of " + config.numEpochs * config.itersPerEpoch + " at " + Logger.time());

			float cost = 0;
			for (int m = 0; m < config.batchesPerIter; m++) {
				//todo build minibatch
				TreeExample tree = forest.getTree();
				while(tree.collapse()) {

				}
				//cost += network.trainMinibatch(config, );
			}
			Logger.log("Average cost per example: " + (cost / config.batchesPerIter) + "\n");

			Logger.log("Validating at " + Logger.time() + "...");
			Logger.log(" training set...");

			Logger.flush();
//			if(config.shouldWriteModel(i)) {
//				network.writeToFile("models/"+logger.timeString+"_"+i+".nnet");
//			}
		}
	}
}
