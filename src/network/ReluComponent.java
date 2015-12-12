package network;

import containers.Matrix;
import tools.Logger;
import tools.TrainConfig;

import java.util.Stack;

/**
 * Created by Andrew on 11/14/2015.
 */
public class ReluComponent implements NetworkComponent {
	private int dim;

	public ReluComponent(int d) {
		dim = d;
	}

	@Override
	public int inputDim() {
		return dim;
	}

	@Override
	public int outputDim() {
		return dim;
	}

	@Override
	public boolean shouldSaveInput() {
		return true;
	}

	@Override
	public boolean shouldSaveError() {
		return false;
	}

	@Override
	public Matrix forward(Matrix input) {
		return input.applyRelu();
	}

	@Override
	public Matrix backward(Matrix error, Matrix input) {
		return error.applyReluPrime(input);
	}

	@Override
	public void update(TrainConfig config, Matrix input, Matrix error) {

	}

	@Override
	public void toString(StringBuilder builder) {
		builder.append("{\n\t\"type\": \"Relu\",");
		builder.append("\n\t\"dim\": ");
		builder.append(dim);
		builder.append("\n}");
	}
}
