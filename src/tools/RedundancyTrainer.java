package tools;

import containers.Dataset;
import containers.Example;
import containers.SentenceDataset;
import edu.berkeley.nlp.lm.ContextEncodedProbBackoffLm;
import edu.berkeley.nlp.lm.io.LmReaders;
import org.json.JSONObject;

import network.NetworkNode;
import readers.RedundancyReader;

import java.util.ArrayList;

/**
 * Created by Andrew on 1/9/2016.
 */
public class RedundancyTrainer {

	private static final int NUM_TRAIN = 46492;

	public static void main(String args[]) {

		Logger.init("log/redundancy");

		/** Read LM **/
		Logger.log("Reading LM at " + Logger.time() + "...");
		ContextEncodedProbBackoffLm<String> lm =
				LmReaders.readContextEncodedLmFromArpa("res/redundancy/lm_giga_5k_nvp_3gram.arpa");

		/** Read train dataset **/
		Logger.log("Reading training data at "+Logger.time()+"...");
		Dataset trainData = new Dataset();
		SentenceDataset trainSentences = new SentenceDataset();
		RedundancyReader.Read(trainData, trainSentences, "res/redundancy/train.txt", lm);

		/** Read test dataset **/
		Logger.log("Reading testing data at "+Logger.time()+"...");
		SentenceDataset testSentences = new SentenceDataset();
		RedundancyReader.Read(new Dataset(), testSentences, "res/redundancy/test.txt", lm);

		/** Read network training config **/
		TrainConfig trainConfig = null;
		try {
			trainConfig = new TrainConfig(
					new JSONObject(Common.readFileIntoString("conf/train_redundancy.config")), NUM_TRAIN);
		} catch(Exception e) {e.printStackTrace();Logger.die("Unable to parse config file: "+args[0]);}
		Logger.log("" + trainConfig);

		NetworkNode network = new NetworkNode(trainConfig.getTopology());

		/** Train network **/
		Logger.log("Beginning training at "+Logger.time()+"...");
		Logger.flush();
		Train(trainConfig, network, trainData, trainSentences, testSentences);

		Logger.log(network.toJsonObject());

		Logger.close();
	}

	public static void Train(TrainConfig config, NetworkNode network,
							 Dataset trainData,
							 SentenceDataset trainSentences,
							 SentenceDataset testSentences) {
		Logger.log("Decoding at " + Logger.time() + "...");
		Logger.log(" training set...");
		Decode(network, trainSentences);
		Logger.log(" test set...");
		Decode(network, testSentences);
		Logger.flush();

		int epoch = 0;
		for (int i = 0; i < config.numEpochs * config.itersPerEpoch; i++) {
			if(i > 0 && i % config.itersBetweenWritingModel == 0) {
				Common.write("models/redundancy/" + Logger.timeString + "." + i + ".dnn", network.toJsonObject());
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

			Logger.log("Decoding at " + Logger.time() + "...");
			Logger.log(" training set...");
			Decode(network, trainSentences);
			Logger.log(" test set...");
			Decode(network, testSentences);

			Logger.flush();
		}
	}

	public static void Decode(NetworkNode network, SentenceDataset sentences) {
		int numCorrect = 0;
		for(int i=0; i<sentences.size(); i++) {
			ArrayList<Example> sentence = sentences.getSentenceExample(i);

			// Actual
			float max = -1f;
			int actual = -1;
			for(int j=0; j<sentence.size(); j++) {
				float[] result = network.forward(sentence.get(j).input, false).getValues();
				if(result[0] > max) {
					max = result[0];
					actual = j;
				}
			}

			// Target
			max = -1f;
			int target = -2;
			for(int j=0; j<sentence.size(); j++) {
				float[] result = sentence.get(j).target.getValues();
				if(result[0] > max) {
					max = result[0];
					target = j;
				}
			}

//			ArrayList<String> words = sentences.getSentenceString(i);
//			for(String w: words) {
//				Logger.log(w+" ", false);
//			}
//			Logger.log("\ntarget: "+words.get(target));
//			Logger.log("actual: "+words.get(actual)+"\n");

			if(actual == target) {
				numCorrect++;
			}
		}
		Logger.log("Accuracy: "+((float)numCorrect*100f/(float)sentences.size()));
	}
}
