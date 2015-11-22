package network;

import containers.Dataset;
import containers.Matrix;
import containers.Minibatch;
import org.json.JSONObject;
import tools.Logger;
import tools.TrainConfig;

import java.util.ArrayList;

/**
 * Created by Andrew on 11/11/2015.
 */
public class NeuralNetwork {
	private ArrayList<NetworkComponent> components;

	public NeuralNetwork(ArrayList<JSONObject> topology) {
		components = new ArrayList<>();
		try {
			for (JSONObject obj : topology) {
				String type = obj.getString("type");
				switch(type) {
					case "Affine":
						components.add(new AffineComponent(
								obj.getInt("inputDim"),
								obj.getInt("outputDim")
						));
						break;
					case "Sigmoid":
						components.add(new SigmoidComponent(
								obj.getInt("dim")
						));
						break;
					case "Tanh":
						components.add(new TanhComponent(
								obj.getInt("dim")
						));
						break;
					case "Relu":
						components.add(new ReluComponent(
								obj.getInt("dim")
						));
						break;
					case "Relu2":
						components.add(new Relu2Component(
								obj.getInt("dim")
						));
						break;
					case "Softmax":
						components.add(new SoftmaxComponent(
								obj.getInt("dim")
						));
						break;
					default:
						Logger.die("Unsupported component type: "+type);
				}
			}
		} catch(Exception e) {e.printStackTrace();}
	}

	public int inputDim() {
		return components.get(0).inputDim();
	}

	public int outputDim() {
		return components.get(components.size()-1).outputDim();
	}

	public float trainMinibatch(TrainConfig config, Minibatch minibatch) {

		Matrix data = minibatch.getInputs();
		data.makeImmutable();

		data = forward(data);

		float cost = calcObjectiveFunction(config, data, minibatch);

		backward(data);

		update(config);

		return cost;
	}

	public Matrix forward(Matrix data) {
		for(NetworkComponent component: components) data = component.forward(data);
		return data;
	}

	public float calcObjectiveFunction(TrainConfig config, Matrix data, Minibatch minibatch) {
		switch (config.objectiveFunction) {
			case TrainConfig.CROSS_ENTROPY:
				return data.applyCrossEntropyError(config, minibatch.getTargets());
			case TrainConfig.MEAN_SQUARED:
				return data.applyMeanSquaredError(config, minibatch.getTargets());
		}
		return -1;
	}

	public float calcObjectiveFunction(TrainConfig config, Matrix data, Matrix targets) {
		switch (config.objectiveFunction) {
			case TrainConfig.CROSS_ENTROPY:
				return data.applyCrossEntropyError(config, targets);
			case TrainConfig.MEAN_SQUARED:
				return data.applyMeanSquaredError(config, targets);
		}
		return -1;
	}

	public Matrix backward(Matrix data) {
		for(int i=components.size()-1; i>=0; i--) data = components.get(i).backward(data);
		return data;
	}

	public void update(TrainConfig config) {
		for(NetworkComponent component: components) component.update(config);
	}

	public String toString() {
		StringBuilder builder = new StringBuilder();
		builder.append("{\n\"NeuralNetwork\": [\n");
		for(NetworkComponent component: components) {
			component.toString(builder);
			builder.append(",\n");
		}
		builder.deleteCharAt(builder.length()-2);
		builder.append("]\n}");
		return builder.toString();
	}

}
