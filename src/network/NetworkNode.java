package network;

import containers.Matrix;
import containers.Minibatch;
import org.json.JSONObject;
import tools.Common;
import tools.Logger;
import tools.TrainConfig;

import java.util.ArrayList;

/**
 * Created by Andrew on 11/11/2015.
 */
public class NetworkNode {
	private ArrayList<NetworkComponent> components;
	private ArrayList<Matrix> inputs;
	private ArrayList<Matrix> errors;

	public NetworkNode(ArrayList<JSONObject> topology) {
		components = new ArrayList<>();
		inputs = new ArrayList<>();
		errors = new ArrayList<>();
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
				inputs.add(null);
				errors.add(null);
			}
		} catch(Exception e) {e.printStackTrace();}
	}

	public NetworkNode(NetworkNode other) {
		components = new ArrayList<>();
		inputs = new ArrayList<>();
		errors = new ArrayList<>();
		components.addAll(other.components);
		inputs.addAll(other.inputs);
		errors.addAll(other.errors);
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

	public Matrix forward(Matrix input) {
		return forward(input, true);
	}

	public Matrix forward(Matrix input, boolean saveInput) {
		for(int i=0; i<components.size(); i++) {
			if(saveInput && components.get(i).shouldSaveInput()) inputs.set(i, new Matrix(input));
			input = components.get(i).forward(input);
		}
		return input;
	}

	public float calcObjectiveFunction(TrainConfig config, Matrix data, Minibatch minibatch) {
		switch (config.objectiveFunction) {
			case Common.CROSS_ENTROPY:
				return data.applyCrossEntropyError(minibatch.getTargets(), minibatch.size());
			case Common.MEAN_SQUARED:
				return data.applyMeanSquaredError(minibatch.getTargets(), minibatch.size());
		}
		return -1;
	}

	public float calcObjectiveFunction(TrainConfig config, Matrix data, Matrix targets) {
		switch (config.objectiveFunction) {
			case Common.CROSS_ENTROPY:
				return data.applyCrossEntropyError(targets, config.minibatchSize);
			case Common.MEAN_SQUARED:
				return data.applyMeanSquaredError(targets, config.minibatchSize);
		}
		return -1;
	}

	public Matrix backward(Matrix error) {
		for(int i=components.size()-1; i>=0; i--) {
			if(components.get(i).shouldSaveError()) {
				error.makeImmutable();
				errors.set(i, error);
			}
			error = components.get(i).backward(error, inputs.get(i));
		}
		return error;
	}

	public void update(TrainConfig config) {
		for(int i=0; i<components.size(); i++) {
			components.get(i).update(config, inputs.get(i), errors.get(i));
		}
	}

	public String toJsonObject() {
		StringBuilder builder = new StringBuilder();
		builder.append("{\n\"NetworkNode\": [\n");
		for(NetworkComponent component: components) {
			component.toString(builder);
			builder.append(",\n");
		}
		builder.deleteCharAt(builder.length()-2);
		builder.append("]\n}");
		return builder.toString();
	}

}
