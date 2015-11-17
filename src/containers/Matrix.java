package containers;

import tools.Logger;
import tools.TrainConfig;

import java.util.Arrays;
import java.util.Random;

/**
 * Created by Andrew on 11/12/2015.
 */
public class Matrix {
	private float[] values;
	private int numRows, numCols, stride;
	private int pointer;
	private boolean isMutable;

	public Matrix(int r, int c) {
		numRows = r;
		numCols = c;
		stride = numCols;
		values = new float[numRows*numCols];
		pointer = 0;
		isMutable = true;
	}

	public Matrix(Matrix other) {
		numRows = other.numRows;
		numCols = other.numCols;
		stride = other.stride;
		values = new float[numRows*numCols];
		pointer = 0;
		isMutable = true;
		System.arraycopy(other.values, 0, this.values, 0, numRows*numCols);
	}

	public void makeImmutable() {
		isMutable = false;
	}

	public void copyRow(Matrix row) {
		if(!isMutable) Logger.die("Tried to change an immutable matrix of dimensions"+numRows+"x"+numCols);
		copyRow(row.values, pointer);
	}

	public void copyRow(float[] row) {
		if(!isMutable) Logger.die("Tried to change an immutable matrix of dimensions"+numRows+"x"+numCols);
		copyRow(row, pointer);
	}

	public void copyRow(float[] row, int p) {
		if(!isMutable) Logger.die("Tried to change an immutable matrix of dimensions"+numRows+"x"+numCols);
		if(row.length != numCols) Logger.die("Tried to copy a row of dim "+row.length+" into a matrix with "+numCols+" columns");
		if(p >= values.length || p % numCols != 0) Logger.die("Invalid pointer supplied to copyRow");
		System.arraycopy(row, 0, values, p, numCols);
		pointer += numCols;
	}

	public void copyRows(float[] rows) {
		if(!isMutable) Logger.die("Tried to change an immutable matrix of dimensions"+numRows+"x"+numCols);
		if(rows.length != numRows * numCols) {
			Logger.die("Tried to copy an array of dim "+rows.length+" into a "+numRows+"x"+numCols+" matrix");
		}
		System.arraycopy(rows, 0, values, 0, numCols);
		pointer = values.length;
	}

	public void randomize() {
		if(!isMutable) Logger.die("Tried to change an immutable matrix of dimensions"+numRows+"x"+numCols);
		Random rand = new Random();
		for(int x=0; x<numRows; x++) {
			for (int y = 0; y < numCols; y++) {
				values[x*stride + y] = (float)(rand.nextGaussian() / Math.sqrt(numCols));
			}
		}
	}

	public Matrix multiply(Matrix other) {
		if(this.numCols != other.numRows) {
			Logger.die("Cannot multiply a "+this.numRows+"x"+this.numCols+
					" matrix with a "+other.numRows+"x"+other.numCols+" matrix");
		}
		Matrix result = new Matrix(this.numRows, other.numCols);
		for(int x=0; x<this.numRows; x++) {
			for(int z = 0; z < other.numCols; z++) {
				for (int y = 0; y < other.numRows; y++) {
					result.values[x*other.stride + z] +=
							this.values[x*this.stride + y] * other.values[y*other.stride + z];
				}
			}
		}
		return result;
	}

	public Matrix multiplyTranspose(Matrix other) {
		if(this.numCols != other.numCols) {
			Logger.die("Cannot multiply a "+this.numRows+"x"+this.numCols+
					" matrix with the transpose of a "+other.numRows+"x"+other.numCols+" matrix");
		}
		Matrix result = new Matrix(this.numRows, other.numRows);
		for(int x=0; x<this.numRows; x++) {
			for (int y = 0; y < this.numCols; y++) {
				for(int z = 0; z < other.numRows; z++) {
					result.values[x*other.stride + z] +=
							this.values[x*this.stride + y] * other.values[z*other.stride + y];
				}
			}
		}
		return result;
	}

	public Matrix transposeMultiply(Matrix other) {
		if(this.numRows != other.numRows) {
			Logger.die("Cannot the transpose of a "+this.numRows+"x"+this.numCols+
					" matrix with a "+other.numRows+"x"+other.numCols+" matrix");
		}
		Matrix result = new Matrix(this.numCols, other.numCols);
		for(int x=0; x<other.numRows; x++) {
			for (int y = 0; y < this.numCols; y++) {
				for(int z = 0; z < other.numCols; z++) {
					result.values[y*other.stride + z] +=
							this.values[x*this.stride + y] * other.values[x*other.stride + z];
				}
			}
		}
		return result;
	}

	public Matrix applyBias(Matrix biases) {
		if(!isMutable) Logger.die("Tried to change an immutable matrix of dimensions"+numRows+"x"+numCols);
		if(biases.numCols != this.numCols) {
			Logger.die("Cannot apply bias vector of dim "+biases.numCols+" to matrix with "+this.numCols+" columns");
		}
		for(int x=0; x<this.numRows; x++) {
			for (int y = 0; y < this.numCols; y++) {
				this.values[x*stride + y] += biases.values[y];
			}
		}
		return this;
	}

	public float applyCrossEntropyError(TrainConfig config, Matrix targets) {
		if(!isMutable) Logger.die("Tried to change an immutable matrix of dimensions"+numRows+"x"+numCols);
		if(this.numRows != targets.numRows || this.numCols != targets.numCols) {
			Logger.die("Cannot calculate cross entropy error between a "+this.numRows+"x"+this.numCols+
					" matrix and a "+targets.numRows+"x"+targets.numCols+" matrix");
		}
		float cost = 0;
		for(int x=0; x<numRows; x++) {
			for (int y=0; y<numCols; y++) {
				cost -= targets.values[x*stride + y] * Math.log(this.values[x*stride + y]);
				this.values[x*stride + y] = targets.values[x*stride + y] - this.values[x*stride + y];
			}
		}
		return cost / config.minibatchSize;
	}

	public float applyMeanSquaredError(TrainConfig config, Matrix targets) {
		if(!isMutable) Logger.die("Tried to change an immutable matrix of dimensions"+numRows+"x"+numCols);
		if(this.numRows != targets.numRows || this.numCols != targets.numCols) {
			Logger.die("Cannot calculate mean squared error between a "+this.numRows+"x"+this.numCols+
					" matrix and a "+targets.numRows+"x"+targets.numCols+" matrix");
		}
		float cost = 0;
		for(int x=0; x<numRows; x++) {
			for (int y=0; y<numCols; y++) {
				cost += (float)Math.pow(this.values[x*stride + y] - targets.values[x*stride + y], 2);
				this.values[x*stride + y] = targets.values[x*stride + y] - this.values[x*stride + y];
			}
		}
		return cost / config.minibatchSize;
	}

	public Matrix applySigmoid() {
		if(!isMutable) Logger.die("Tried to change an immutable matrix of dimensions"+numRows+"x"+numCols);
		for(int x=0; x<numRows; x++) {
			for (int y = 0; y < numCols; y++) {
				values[x*stride + y] = sigmoid(values[x * stride + y]);
			}
		}
		return this;
	}

	public Matrix applySigmoidPrime(Matrix other) {
		if(!isMutable) Logger.die("Tried to change an immutable matrix of dimensions"+numRows+"x"+numCols);
		if(this.numRows != other.numRows || this.numCols != other.numCols) {
			Logger.die("Cannot calculate hadamard product between a "+this.numRows+"x"+this.numCols+
					" matrix and a "+other.numRows+"x"+other.numCols+" matrix");
		}
		for(int x=0; x<numRows; x++) {
			for (int y=0; y<numCols; y++) {
				this.values[x*stride + y] = this.values[x*stride + y] * sigmoidPrime(other.values[x*stride + y]);
			}
		}
		return this;
	}

	public Matrix applyTanh() {
		if(!isMutable) Logger.die("Tried to change an immutable matrix of dimensions"+numRows+"x"+numCols);
		for(int x=0; x<numRows; x++) {
			for (int y = 0; y < numCols; y++) {
				values[x*stride + y] = tanh(values[x * stride + y]);
			}
		}
		return this;
	}

	public Matrix applyTanhPrime(Matrix other) {
		if(!isMutable) Logger.die("Tried to change an immutable matrix of dimensions"+numRows+"x"+numCols);
		if(this.numRows != other.numRows || this.numCols != other.numCols) {
			Logger.die("Cannot calculate hadamard product between a "+this.numRows+"x"+this.numCols+
					" matrix and a "+other.numRows+"x"+other.numCols+" matrix");
		}
		for(int x=0; x<numRows; x++) {
			for (int y=0; y<numCols; y++) {
				this.values[x*stride + y] = this.values[x*stride + y] * tanhPrime(other.values[x * stride + y]);
			}
		}
		return this;
	}

	public Matrix applyRelu() {
		if(!isMutable) Logger.die("Tried to change an immutable matrix of dimensions"+numRows+"x"+numCols);
		for(int x=0; x<numRows; x++) {
			for (int y = 0; y < numCols; y++) {
				values[x*stride + y] = relu(values[x * stride + y]);
			}
		}
		return this;
	}

	public Matrix applyReluPrime(Matrix other) {
		if(!isMutable) Logger.die("Tried to change an immutable matrix of dimensions"+numRows+"x"+numCols);
		if(this.numRows != other.numRows || this.numCols != other.numCols) {
			Logger.die("Cannot calculate hadamard product between a "+this.numRows+"x"+this.numCols+
					" matrix and a "+other.numRows+"x"+other.numCols+" matrix");
		}
		for(int x=0; x<numRows; x++) {
			for (int y=0; y<numCols; y++) {
				this.values[x*stride + y] = this.values[x*stride + y] * reluPrime(other.values[x * stride + y]);
			}
		}
		return this;
	}

	public Matrix applyRelu2() {
		if(!isMutable) Logger.die("Tried to change an immutable matrix of dimensions"+numRows+"x"+numCols);
		for(int x=0; x<numRows; x++) {
			for (int y = 0; y < numCols; y++) {
				values[x*stride + y] = relu2(values[x * stride + y]);
			}
		}
		return this;
	}

	public Matrix applyRelu2Prime(Matrix other) {
		if(!isMutable) Logger.die("Tried to change an immutable matrix of dimensions"+numRows+"x"+numCols);
		if(this.numRows != other.numRows || this.numCols != other.numCols) {
			Logger.die("Cannot calculate hadamard product between a "+this.numRows+"x"+this.numCols+
					" matrix and a "+other.numRows+"x"+other.numCols+" matrix");
		}
		for(int x=0; x<numRows; x++) {
			for (int y=0; y<numCols; y++) {
				this.values[x*stride + y] = this.values[x*stride + y] * relu2Prime(other.values[x * stride + y]);
			}
		}
		return this;
	}

	public Matrix applySoftmax() {
		if(!isMutable) Logger.die("Tried to change an immutable matrix of dimensions"+numRows+"x"+numCols);
		for(int x=0; x<numRows; x++) {
			float sum = 0;
			for (int y = 0; y < numCols; y++) {
				values[x*stride + y] = (float)Math.exp(values[x*stride + y]);
				sum += values[x*stride + y];
			}
			for (int y = 0; y < numCols; y++) {
				values[x*stride + y] /= sum;
			}
		}
		return this;
	}

	public void updateBias(Matrix delta, float learningFactor) {
		if(!isMutable) Logger.die("Tried to change an immutable matrix of dimensions"+numRows+"x"+numCols);
		if(this.numCols != delta.numCols) {
			Logger.die("Cannot update bias vector of dim "+this.numCols+" by a matrix with "+delta.numCols+" columns");
		}
		for(int x=0; x<delta.numRows; x++) {
			for (int y = 0; y < delta.numCols; y++) {
				this.values[y] -= delta.values[x * delta.stride + y] * learningFactor;
			}
		}
	}

	public void updateWeights(Matrix delta, float learningFactor, float decayFactor) {
		if(!isMutable) Logger.die("Tried to change an immutable matrix of dimensions"+numRows+"x"+numCols);
		for(int x=0; x<numRows; x++) {
			for (int y=0; y<numCols; y++) {
				this.values[x*stride + y] = this.values[x*stride + y] * decayFactor +
					delta.values[x*stride + y] * learningFactor;
			}
		}
	}

	public int getIndexOfMaxElement() {
		float max = -1;
		int indexOfMax = -1;
		for(int x=0; x<numRows; x++) {
			for (int y = 0; y < numCols; y++) {
				int index = x*stride + y;
				if(values[index] > max) {
					max = values[index];
					indexOfMax = index;
				}
			}
		}
		return indexOfMax;
	}

	public void toString(StringBuilder builder){
		for(int x=0; x<numRows; x++) {
			builder.append("\t\t");
			for (int y = 0; y < numCols; y++) {
				builder.append(values[x*stride + y]);
				builder.append(",");
			}
			builder.append("\n");
		}
		builder.deleteCharAt(builder.length()-2);
	}

	private float sigmoid(float f) {
		return (float)( 1 / (1 + Math.exp(-f)) );
	}

	private float sigmoidPrime(float f) {
		double exp = Math.exp(-f);
		return (float)( (1 - 1 / (1 + exp)) * (1 / (1 + exp)) );
	}

	private float tanh(float f) {
		double exp = Math.exp(f);
		return (float)( ( exp - 1/exp ) / (exp + 1/exp) );
	}

	private float tanhPrime(float f) {
		double exp = Math.exp(2*f);
		return (float)( 4 / (exp + 1/exp + 2) );
	}

	private float relu(float f) {
		return f > 0 ? f : 0;
	}

	private float reluPrime(float f) {
		return f > 0 ? 1 : 0;
	}

	private float relu2(float f) {
		return f > 0 ? (f < 1 ? f : 1 ) : 0;
	}

	private float relu2Prime(float f) {
		return f > 0 ? (f < 1 ? 1 : 0 ) : 0;
	}
}