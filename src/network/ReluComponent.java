package network;

import containers.Matrix;
import tools.Logger;
import tools.TrainConfig;

/**
 * Created by Andrew on 11/14/2015.
 */
public class ReluComponent implements NetworkComponent {
	private Matrix input;

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
	public Matrix forward(Matrix matrix) {
		this.input = new Matrix(input);
		return input.applyRelu();
	}

	@Override
	public Matrix backward(Matrix matrix) {
		if(input == null) Logger.die("Tried to backprop on a ReluComponent that has not received input");
		return matrix.applyReluPrime(input);
	}

	@Override
	public void update(TrainConfig config) {
		input = null;
	}

	@Override
	public void toString(StringBuilder builder) {
		builder.append("{\n\t\"type\": \"Relu\",");
		builder.append("\n\t\"dim\": ");
		builder.append(dim);
		builder.append("\n}");
	}
}
