package containers;

/**
 * Created by Andrew on 11/22/2015.
 */
public class MvPair {
	public Matrix matrix;
	public Matrix vector;

	public Matrix input;

	public MvPair(Matrix v, Matrix m) {
		vector = v;
		matrix = m;
	}
}
