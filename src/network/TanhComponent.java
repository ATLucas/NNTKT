package network;

import containers.Matrix;
import tools.Logger;
import tools.TrainConfig;

/**
 * Created by Andrew on 11/14/2015.
 */
public class TanhComponent implements NetworkComponent {
	private Matrix input;

	private int dim;

	public TanhComponent(int d) {
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
	public Matrix forward(Matrix input) {
		this.input = new Matrix(input);
		return input.applyTanh();
	}

	@Override
	public Matrix backward(Matrix matrix) {
		if(input == null) Logger.die("Tried to backprop on a SigmoidComponent that has not received input");
		return matrix.applyTanhPrime(input);
	}

	@Override
	public void update(TrainConfig config) {
		input = null;
	}

	@Override
	public void toString(StringBuilder builder) {
		builder.append("{\n\t\"type\": \"Tanh\",");
		builder.append("\n\t\"dim\": ");
		builder.append(dim);
		builder.append("\n}");
	}
}
