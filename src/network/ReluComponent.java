package network;

import containers.Matrix;
import tools.Logger;
import tools.TrainConfig;

import java.util.Stack;

/**
 * Created by Andrew on 11/14/2015.
 */
public class ReluComponent implements NetworkComponent {
	private Stack<Matrix> inputs;

	private int dim;

	public ReluComponent(int d) {
		dim = d;
		inputs = new Stack<>();
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
		inputs.push(new Matrix(matrix));
		return matrix.applyRelu();
	}

	@Override
	public Matrix backward(Matrix matrix) {
		Matrix input = inputs.pop();
		if(input == null) Logger.die("Tried to backprop on a ReluComponent that has not received input");
		return matrix.applyReluPrime(input);
	}

	@Override
	public void update(TrainConfig config) {}

	@Override
	public void toString(StringBuilder builder) {
		builder.append("{\n\t\"type\": \"Relu\",");
		builder.append("\n\t\"dim\": ");
		builder.append(dim);
		builder.append("\n}");
	}
}
