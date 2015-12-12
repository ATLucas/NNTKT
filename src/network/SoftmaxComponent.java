package network;

import containers.Matrix;
import tools.TrainConfig;

/**
 * Created by Andrew on 11/14/2015.
 */
public class SoftmaxComponent implements NetworkComponent {
	private int dim;

	public SoftmaxComponent(int d) {
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
		return false;
	}

	@Override
	public boolean shouldSaveError() {
		return false;
	}

	@Override
	public Matrix forward(Matrix input) {
		return input.applySoftmax();
	}

	@Override
	public Matrix backward(Matrix error, Matrix input) {
		return error;
	}

	@Override
	public void update(TrainConfig config, Matrix input, Matrix error) {

	}

	@Override
	public void toString(StringBuilder builder) {
		builder.append("{\n\t\"type\": \"Softmax\",");
		builder.append("\n\t\"dim\": ");
		builder.append(dim);
		builder.append("\n}");
	}
}
