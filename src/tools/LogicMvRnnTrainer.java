package tools;

import containers.MvRnnForest;
import containers.Matrix;
import containers.MvRnnTree;
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
			vocab = new Vocab(new JSONObject(Common.readFileIntoString(args[1])));
		} catch(Exception e){e.printStackTrace();}

		MvRnnForest mvRnnForest = new MvRnnForest();
		ForestReader.Read(mvRnnForest, vocab, args[2]);

		Train(trainVectorConfig, mvRnnForest, vocab.wordDim());

		Logger.close();
	}

	public static void Train(TrainConfig config, MvRnnForest mvRnnForest, int wordDim) {
		NeuralNetwork network = new NeuralNetwork(config.getTopology());
		Matrix weights = new Matrix(wordDim*2,wordDim);
		weights.randomize();

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
				MvRnnTree mvRnnTree = mvRnnForest.getTree();
				cost += mvRnnTree.train(config, network, weights);
			}
			Logger.log("Average cost per example: " + (cost / config.batchesPerIter) + "\n");
			Logger.flush();
		}

		Logger.log("Validating at " + Logger.time() + "...");
		Logger.log(" training set...");
		for (int j = 0; j < mvRnnForest.size(); j++) {
			MvRnnTree mvRnnTree = mvRnnForest.getTree(j);
			Logger.log("\ntarget vector: " + mvRnnTree.getTopLabel().vector.toString());
			Logger.log("actual vector: "+ mvRnnTree.decode(network, weights).vector.toString());
			Logger.log("target matrix: " + mvRnnTree.getTopLabel().matrix.toString());
			Logger.log("actual matrix: " + mvRnnTree.decode(network, weights).matrix.toString());
		}

		Common.write("models/logic/" + Logger.timeString + "." + "final.v.dnn", network.toString());
	}
}
