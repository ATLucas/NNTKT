package nlp;

import containers.Matrix;
import containers.MvPair;

/**
 * Created by Andrew on 11/15/2015.
 */
public class Word {
	private int id, dim;
	private boolean updatable;
	public MvPair mvPair;

	public Word(int dim, int id, float[] vec, float[] mat, boolean updatable){
		this.dim = dim;
		this.id = id;
		this.updatable = updatable;
		Matrix vector = new Matrix(1, dim);
		Matrix matrix = new Matrix(dim, dim);
		vector.copyRow(vec);
		matrix.copyRows(mat);
		mvPair = new MvPair(vector, matrix);
	}

	public int id() {
		return id;
	}

	public int dim() {
		return dim;
	}

	public boolean isUpdatable() {
		return updatable;
	}

	public void update(Matrix v, Matrix m, float vectorLearningRate, float matrixLearningRate) {
		if(updatable) {
			mvPair.vector.update(v, vectorLearningRate);
			mvPair.matrix.update(m, matrixLearningRate);
		}
	}
}
