package containers;

/**
 * Created by Andrew on 11/22/2015.
 */
public class MvPair {
	public Matrix matrix;
	public Matrix vector;

	public MvPair(Matrix v, Matrix m) {
		vector = v;
		matrix = m;
	}

	public MvPair(MvPair other) {
		vector = new Matrix(other.vector);
		matrix = new Matrix(other.matrix);
	}
}
