package network;

import containers.Matrix;
import tools.Logger;
import tools.TrainConfig;

import java.util.Stack;

/**
 * Created by Andrew on 11/13/2015.
 */
public class AffineComponent implements NetworkComponent {
	private Matrix weights;
	private Matrix biases;
	private Stack<Matrix> inputs;
	private Stack<Matrix> errors;

	private int inputDim, outputDim;

	public AffineComponent(int r, int c) {
		inputDim = r;
		outputDim = c;

		weights = new Matrix(r,c);
		biases = new Matrix(1,c);

		weights.randomize();
		biases.randomize();

		inputs = new Stack<>();
		errors = new Stack<>();
	}

	@Override
	public int inputDim() {
		return inputDim;
	}

	@Override
	public int outputDim() {
		return outputDim;
	}

	@Override
	public Matrix forward(Matrix input) {
		input.makeImmutable();
		inputs.push(input);
		return input.multiply(weights).applyBias(biases);
	}

	@Override
	public Matrix backward(Matrix error) {
		error.makeImmutable();
		errors.push(error);
		return error.multiplyTranspose(weights);
	}

	@Override
	public void update(TrainConfig config) {
		if(inputs.size() == 0) Logger.die("Tried to update an AffineComponent that has not received input");
		if(errors.size() == 0) Logger.die("Tried to update an AffineComponent that has not received error");
		Matrix input = inputs.pop();
		Matrix error = errors.pop();
		biases.updateBias(error, config.learningRate / config.minibatchSize);
		weights.updateWeights(input.transposeMultiply(error),
				config.learningRate / config.minibatchSize,
				config.decayFactor);
	}

	@Override
	public void toString(StringBuilder builder) {
		builder.append("{\n\t\"type\": Affine");
		builder.append(",\n\t\"inputDim\": ");
		builder.append(inputDim);
		builder.append(",\n\t\"outputDim\": ");
		builder.append(outputDim);
		builder.append(",\n\t\"weights\": [\n");
		weights.toString(builder);
		builder.append("\t],\n\t\"biases\": [\n");
		biases.toString(builder);
		builder.append("\t]\n}");
	}
}
