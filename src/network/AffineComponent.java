package network;

import containers.Matrix;
import tools.Logger;
import tools.TrainConfig;

/**
 * Created by Andrew on 11/13/2015.
 */
public class AffineComponent implements NetworkComponent {
	private Matrix weights;
	private Matrix biases;
	private Matrix input;
	private Matrix error;

	private int inputDim, outputDim;

	public AffineComponent(int r, int c) {
		inputDim = r;
		outputDim = c;

		weights = new Matrix(r,c);
		biases = new Matrix(1,c);
		error = null;

		weights.randomize();
		biases.randomize();
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
		this.input = input;
		this.input.makeImmutable();
		return input.multiply(weights).applyBias(biases);
	}

	@Override
	public Matrix backward(Matrix error) {
		this.error = error;
		this.error.makeImmutable();
		return error.multiplyTranspose(weights);
	}

	@Override
	public void update(TrainConfig config) {
		if(input == null) Logger.die("Tried to update an AffineComponent that has not received input");
		if(error == null) Logger.die("Tried to update an AffineComponent that has not received error");
		biases.updateBias(error, config.learningRate / config.minibatchSize);
		weights.updateWeights(input.transposeMultiply(error),
				config.learningRate / config.minibatchSize,
				config.decayFactor);
		input = null;
		error = null;
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
