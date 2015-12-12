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

	private int inputDim, outputDim;

	public AffineComponent(int r, int c) {
		inputDim = r;
		outputDim = c;

		weights = new Matrix(r,c);
		biases = new Matrix(1,c);

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
	public boolean shouldSaveInput() {
		return true;
	}

	@Override
	public boolean shouldSaveError() {
		return true;
	}

	@Override
	public Matrix forward(Matrix input) {
		return input.multiply(weights).applyBias(biases);
	}

	@Override
	public Matrix backward(Matrix error, Matrix input) {
		return error.multiplyTranspose(weights);
	}

	@Override
	public void update(TrainConfig config, Matrix input, Matrix error) {
		biases.updateBias(error, config.learningRate / config.minibatchSize);
		weights.updateWeights(input.transposeMultiply(error),
				config.learningRate / config.minibatchSize,
				config.decayFactor);
	}

	@Override
	public void toString(StringBuilder builder) {
		builder.append("{\n  \"type\": \"Affine\"");
		builder.append(",\n  \"inputDim\": ");
		builder.append(inputDim);
		builder.append(",\n  \"outputDim\": ");
		builder.append(outputDim);
		builder.append(",\n  \"weights\": [\n");
		weights.appendSelf(builder, "    ");
		builder.append("  ],\n  \"biases\": [\n");
		biases.appendSelf(builder, "    ");
		builder.append("  ]\n}");
	}
}
