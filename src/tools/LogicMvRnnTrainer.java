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

		MvRnnTrainConfig config = null;
		try {
			config = new MvRnnTrainConfig(new JSONObject(Common.readFileIntoString(args[0])), 6);
		} catch(Exception e){e.printStackTrace();Logger.die("Unable to parse config: "+args[0]);}
		Logger.log("Config:\n" + config);

		Vocab vocab = null;
		try {
			vocab = new Vocab(new JSONObject(Common.readFileIntoString(args[1])));
		} catch(Exception e){e.printStackTrace();}

		MvRnnForest mvRnnForest = new MvRnnForest();
		ForestReader.Read(mvRnnForest, vocab, args[2]);

		Train(config, mvRnnForest, vocab.wordDim());

		Logger.close();
	}

	public static void Train(MvRnnTrainConfig config, MvRnnForest mvRnnForest, int wordDim) {
		NeuralNetwork network = new NeuralNetwork(config.networkConfig.getTopology());
		Matrix weights = new Matrix(wordDim*2,wordDim);
		weights.randomize();

		int epoch = 0;
		for (int i = 0; i < config.networkConfig.numEpochs * config.networkConfig.itersPerEpoch; i++) {
			if (i > 0 && i % config.networkConfig.itersBetweenWritingModel == 0) {
				Common.write("models/logic/" + Logger.timeString + "." + i + ".v.dnn", network.toString());
			}

			if (i % config.networkConfig.itersPerEpoch == 0) {
				Logger.log("\nEpoch " + (++epoch) + " of " + config.networkConfig.numEpochs);
			}
			Logger.log("\nIteration " + (i + 1) + " of " +
					config.networkConfig.numEpochs * config.networkConfig.itersPerEpoch + " at " + Logger.time());

			float cost = 0;
			for (int m = 0; m < config.networkConfig.batchesPerIter; m++) {
				MvRnnTree mvRnnTree = mvRnnForest.getTree();
				cost += mvRnnTree.train(config, network, weights);
			}
			Logger.log("Average cost per example: " + (cost / config.networkConfig.batchesPerIter) + "\n");
			Logger.flush();
		}

		Logger.log("Validating at " + Logger.time() + "...");
		Logger.log(" training set...");
		for (int j = 0; j < mvRnnForest.size(); j++) {
			MvRnnTree mvRnnTree = mvRnnForest.getTree(j);
			Logger.log("\ntarget vector: " + mvRnnTree.getTopLabel().vector.toString());
			Logger.log("actual vector: " + mvRnnTree.decode(network, weights).vector.toString());
			Logger.log("target matrix: " + mvRnnTree.getTopLabel().matrix.toString());
			Logger.log("actual matrix: " + mvRnnTree.decode(network, weights).matrix.toString());
		}

		Common.write("models/logic/" + Logger.timeString + "." + "final.v.dnn", network.toString());
	}
}
