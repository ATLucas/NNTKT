package network;

import containers.Matrix;
import tools.TrainConfig;

import java.util.Stack;

/**
 * Created by Andrew on 12/6/2015.
 */
public class BottleneckComponent implements NetworkComponent {
	private int dim;

	private Stack<Matrix> data;

	public BottleneckComponent(int d) {
		dim = d;
		data = new Stack<>();
	}

	@Override
	public int inputDim() {
		return dim;
	}

	@Override
	public int outputDim() {
		return dim;
	}

	public Matrix getData() {
		return data.pop();
	}

	@Override
	public Matrix forward(Matrix matrix) {
		data.push(new Matrix(matrix));
		return matrix;
	}

	@Override
	public Matrix backward(Matrix matrix) {
		return matrix;
	}

	@Override
	public void update(TrainConfig config) {

	}

	@Override
	public void toString(StringBuilder builder) {
		builder.append("{\n\t\"type\": \"Bottleneck\",");
		builder.append("\n\t\"dim\": ");
		builder.append(dim);
		builder.append("\n}");
	}
}
