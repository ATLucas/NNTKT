package containers;

import java.util.ArrayList;

/**
 * Created by Andrew on 11/12/2015.
 */
public class Minibatch {
	ArrayList<Example> examples;

	public Minibatch() {
		examples = new ArrayList<>();
	}

	public void add(Example example) {
		examples.add(example);
	}

	public int size() {
		return examples.size();
	}

	public Matrix getInputs() {
		Matrix inputs = new Matrix(size(), examples.get(0).inputDim());
		for(Example example: examples) {
			inputs.copyRow(example.input);
		}
		return inputs;
	}

	public Matrix getTargets() {
		Matrix targets = new Matrix(size(), examples.get(0).targetDim());
		for(Example example: examples) {
			targets.copyRow(example.target);
		}
		return targets;
	}

}
