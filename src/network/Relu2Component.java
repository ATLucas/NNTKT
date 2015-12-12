package network;

import containers.Matrix;
import tools.Logger;
import tools.TrainConfig;

import java.util.Stack;

/**
 * Created by Andrew on 11/14/2015.
 */
public class Relu2Component implements NetworkComponent {
	private int dim;

	public Relu2Component(int d) {
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
		return input.applyRelu2();
	}

	@Override
	public Matrix backward(Matrix error, Matrix input) {
		return error.applyRelu2Prime(input);
	}

	@Override
	public void update(TrainConfig config, Matrix input, Matrix error) {

	}

	@Override
	public void toString(StringBuilder builder) {
		builder.append("{\n\t\"type\": \"Relu2\",");
		builder.append("\n\t\"dim\": ");
		builder.append(dim);
		builder.append("\n}");
	}
}
