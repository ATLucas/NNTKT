package nlp;

import containers.Matrix;
import containers.MvPair;

/**
 * Created by Andrew on 11/15/2015.
 */
public class Word {
	public MvPair mvPair;
	protected boolean shouldUpdate;
	private int id, dim;

	public Word(int dim, int id, float[] vec, float[] mat, boolean shouldUpdate){
		this.dim = dim;
		this.id = id;
		this.shouldUpdate = shouldUpdate;
		Matrix vector = new Matrix(1, dim);
		Matrix matrix = new Matrix(dim, dim);
		vector.copyRow(vec);
		matrix.copyRows(mat);
		mvPair = new MvPair(vector, matrix);
	}

	public void update(Matrix v, Matrix m, float vectorLearningRate, float matrixLearningRate) {
		if(shouldUpdate) {
			mvPair.vector.update(v, vectorLearningRate);
			mvPair.matrix.update(m, matrixLearningRate);
		}
	}
}
