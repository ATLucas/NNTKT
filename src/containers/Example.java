package containers;

/**
 * Created by Andrew on 11/11/2015.
 */
public class Example {
	public Matrix input;
	public Matrix target;
	private int inputDim, targetDim;

	public Example(float[] in, float[] tgt) {
		inputDim = in.length;
		targetDim = tgt.length;
		input = new Matrix(1, in.length);
		target = new Matrix(1, tgt.length);
		input.copyRow(in);
		target.copyRow(tgt);
	}

	public int inputDim() {
		return inputDim;
	}

	public int targetDim() {
		return targetDim;
	}

}
