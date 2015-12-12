package tools;

import containers.NetworkForest;
import containers.Matrix;
import containers.NetworkTree;
import network.NetworkNode;
import nlp.Vocab;
import org.json.JSONObject;
import readers.ForestReader;

/**
 * Created by Andrew on 11/15/2015.
 */
public class LogicMvRnnTrainer {

	public static void main(String args[]) {
		Logger.init("log/logic");
		if(args.length != 4) Logger.die("Wrong number of arguments supplied to LogicMvRnnTrainer");

		MvRnnTrainConfig config = null;
		try {
			config = new MvRnnTrainConfig(new JSONObject(Common.readFileIntoString(args[0])), 6);
		} catch(Exception e){e.printStackTrace();Logger.die("Unable to read config: "+args[0]);}
		Logger.log("Config:\n" + config);

		Vocab vocab = null;
		try {
			vocab = new Vocab(new JSONObject(Common.readFileIntoString(args[1])));
		} catch(Exception e){e.printStackTrace();Logger.die("Unable to read vocab: " + args[1]);}
		Logger.log("Vocab:\n" + vocab.toJsonObject());

		NetworkNode network = new NetworkNode(config.networkConfig.getTopology());
		Matrix weights = new Matrix(vocab.wordDim()*2,vocab.wordDim());
		weights.randomize();

		NetworkForest trainForest = new NetworkForest();
		ForestReader.Read(trainForest, vocab, args[2], network, weights);
		Train(config, trainForest, network, weights);

		NetworkForest testForest = new NetworkForest();
		ForestReader.Read(testForest, vocab, args[3], network, weights);
		Logger.log("\nDECODING AT " + Logger.time() + "...");
		Logger.log("\n TRAIN SET...");
		Decode(trainForest);
		Logger.log("\n TEST SET...");
		Decode(testForest);

		Common.write("vocab/logic/" + Logger.timeString + "." + "final.vocab", vocab.toJsonObject());
		Common.write("models/logic/" + Logger.timeString + "." + "final.v.nn", network.toJsonObject());
		Common.write("models/logic/" + Logger.timeString + "." + "final.w.nn",
				"{\n\"weights\":\n"+weights.toJsonObject("  ")+"}"
		);

		Logger.close();
	}

	public static void Train(MvRnnTrainConfig config, NetworkForest networkForest, NetworkNode network, Matrix weights) {


		int epoch = 0;
		for (int i = 0; i < config.networkConfig.numEpochs * config.networkConfig.itersPerEpoch; i++) {
			if (i > 0 && i % config.networkConfig.itersBetweenWritingModel == 0) {
				Common.write("models/logic/" + Logger.timeString + "." + i + ".v.dnn", network.toJsonObject());
			}

			if (i % config.networkConfig.itersPerEpoch == 0) {
				Logger.log("\nEpoch " + (++epoch) + " of " + config.networkConfig.numEpochs);
			}
			Logger.log("\nIteration " + (i + 1) + " of " +
					config.networkConfig.numEpochs * config.networkConfig.itersPerEpoch + " at " + Logger.time());

			float cost = 0;
			for (int m = 0; m < config.networkConfig.batchesPerIter; m++) {
				NetworkTree networkTree = networkForest.getTree();
				cost += networkTree.train(config);
			}
			Logger.log("Average cost per example: " + (cost / config.networkConfig.batchesPerIter) + "\n");
			Logger.flush();
		}
	}

	public static void Decode(NetworkForest networkForest) {
		for (int j = 0; j < networkForest.size(); j++) {
			NetworkTree networkTree = networkForest.getTree(j);
			Logger.log("\ntarget vector: " + networkTree.getTopLabel().vector.toString());
			Logger.log("actual vector: " + networkTree.decode().vector.toString());
			Logger.log("target matrix: " + networkTree.getTopLabel().matrix.toString());
			Logger.log("actual matrix: " + networkTree.decode().matrix.toString());
		}
	}
}
