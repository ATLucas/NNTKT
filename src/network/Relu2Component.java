package network;

import containers.Matrix;
import tools.Logger;
import tools.TrainConfig;

/**
 * Created by Andrew on 11/14/2015.
 */
public class Relu2Component implements NetworkComponent {
	private Matrix input;

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
	public Matrix forward(Matrix matrix) {
		this.input = new Matrix(input);
		return input.applyRelu2();
	}

	@Override
	public Matrix backward(Matrix matrix) {
		if(input == null) Logger.die("Tried to backprop on a Relu2Component that has not received input");
		return matrix.applyRelu2Prime(input);
	}

	@Override
	public void update(TrainConfig config) {
		input = null;
	}

	@Override
	public void toString(StringBuilder builder) {
		builder.append("{\n\t\"type\": \"Relu2\",");
		builder.append("\n\t\"dim\": ");
		builder.append(dim);
		builder.append("\n}");
	}
}
