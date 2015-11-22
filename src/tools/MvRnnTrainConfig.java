package tools;

import org.json.JSONException;
import org.json.JSONObject;

/**
 * Created by Andrew on 11/22/2015.
 */
public class MvRnnTrainConfig {

	public TrainConfig networkConfig;

	public float vectorLearningRate, matrixLearningRate, weightsLearningRate, weightsLambda, weightsDecayFactor;
	public int objectiveFunction;

	private String configFile;

	public MvRnnTrainConfig(JSONObject jsonObject, int numTrainingExamples) throws JSONException {
		configFile = jsonObject.toString(1);

		vectorLearningRate = (float)jsonObject.getDouble("vectorLearningRate");
		matrixLearningRate = (float)jsonObject.getDouble("matrixLearningRate");

		networkConfig = new TrainConfig(jsonObject.getJSONObject("vectorCompositionNetwork"), numTrainingExamples);

		JSONObject weightsParams = jsonObject.getJSONObject("matrixCompositionWeights");
		weightsLearningRate = (float)weightsParams.getDouble("learningRate");
		weightsLambda = (float)weightsParams.getDouble("lambda");
		weightsDecayFactor = 1 - weightsLearningRate * weightsLambda / numTrainingExamples;

		String objFunction = weightsParams.getString("objectiveFunction");
		switch (objFunction) {
			case "CROSS-ENTROPY":
				objectiveFunction = Common.CROSS_ENTROPY;
				break;
			case "MEAN-SQUARED":
				objectiveFunction = Common.MEAN_SQUARED;
				break;
			default:
				Logger.die("Unsupported objective function: " + objFunction);
		}
	}

	public String toString() {
		return "****************************************\n"+configFile+"\n****************************************\n";
	}
}
