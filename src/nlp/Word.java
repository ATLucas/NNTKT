package nlp;

import containers.Matrix;

/**
 * Created by Andrew on 11/15/2015.
 */
public class Word {
	private int id, dim;
	private boolean updatable;
	public Matrix vector;
	public Matrix matrix;

	public Word(int dim, int id, float[] vec, float[] mat, boolean updatable){
		this.dim = dim;
		this.id = id;
		this.updatable = updatable;
		vector = new Matrix(1, dim);
		matrix = new Matrix(dim, dim);
		vector.copyRow(vec);
		matrix.copyRows(mat);
	}

	public int id() {
		return id;
	}

	public int dim() {
		return dim;
	}
}
