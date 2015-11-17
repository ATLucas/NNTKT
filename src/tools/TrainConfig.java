package tools;

import org.json.JSONArray;
import org.json.JSONObject;

import java.nio.charset.Charset;
import java.util.ArrayList;

/**
 * Created by Andrew on 11/12/2015.
 */
public class TrainConfig {
	public static final int CROSS_ENTROPY=1, MEAN_SQUARED=2;

	public int numEpochs, itersPerEpoch, minibatchSize, batchesPerIter, itersBetweenWritingModel, objectiveFunction;
	public float learningRate, lambda, decayFactor;

	public ArrayList<JSONObject> topology;

	private String configFile;

	public TrainConfig(String configFileName, int numTrainingExamples) {
		try {
			JSONObject jsonObject = new JSONObject(
					configFile = Common.readFileIntoString(configFileName));

			JSONObject params = jsonObject.getJSONObject("params");
			numEpochs = params.getInt("numEpochs");
			itersPerEpoch = params.getInt("itersPerEpoch");
			minibatchSize = params.getInt("minibatchSize");
			itersBetweenWritingModel = params.getInt("itersBetweenWritingModel");

			if(numTrainingExamples == -1) numTrainingExamples = params.getInt("numTrainingExamples");

			batchesPerIter = numTrainingExamples / (minibatchSize * itersPerEpoch);
			learningRate = (float)params.getDouble("learningRate");
			lambda = (float)params.getDouble("lambda");
			decayFactor = 1 - learningRate * lambda / numTrainingExamples;

			String objFunction = params.getString("objectiveFunction");
			switch (objFunction) {
				case "CROSS-ENTROPY":
					objectiveFunction = CROSS_ENTROPY;
					break;
				case "MEAN-SQUARED":
					objectiveFunction = MEAN_SQUARED;
					break;
				default:
					Logger.die("Unsupported objective function: " + objFunction);
			}

			JSONArray jsonTopo = jsonObject.getJSONArray("topology");
			topology = new ArrayList<>();
			for(int i=0; i<jsonTopo.length(); i++) {
				topology.add(jsonTopo.getJSONObject(i));
			}

		} catch (Exception e) {
			e.printStackTrace();
			Logger.die("Failed to parse config file: "+configFileName);
		}
	}

	public ArrayList<JSONObject> getTopology() {
		return topology;
	}

	public String toString() {
		return "****************************************\n"+configFile+"\n****************************************\n";
	}
}
