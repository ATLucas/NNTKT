package network;

import containers.Matrix;
import tools.Logger;
import tools.TrainConfig;

import java.util.Stack;

/**
 * Created by Andrew on 11/13/2015.
 */
public class SigmoidComponent implements NetworkComponent {
	private Stack<Matrix> inputs;
	private int dim;

	public SigmoidComponent(int d) {
		inputs = new Stack<>();
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
		inputs.push(new Matrix(matrix));
		return matrix.applySigmoid();
	}

	@Override
	public Matrix backward(Matrix matrix) {
		Matrix input = inputs.pop();
		if(input == null) Logger.die("Tried to backprop on a SigmoidComponent that has not received input");
		return matrix.applySigmoidPrime(input);
	}

	@Override
	public void update(TrainConfig config) {}

	@Override
	public void toString(StringBuilder builder) {
		builder.append("{\n\t\"type\": \"Sigmoid\",");
		builder.append("\n\t\"dim\": ");
		builder.append(dim);
		builder.append("\n}");
	}
}
