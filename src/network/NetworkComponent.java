package network;

import containers.Matrix;
import tools.TrainConfig;

/**
 * Created by Andrew on 11/12/2015.
 */
public interface NetworkComponent {

	int inputDim();
	int outputDim();
	boolean shouldSaveInput();
	boolean shouldSaveError();
	Matrix forward(Matrix input);
	Matrix backward(Matrix error, Matrix input);
	void update(TrainConfig config, Matrix input, Matrix error);
	void toString(StringBuilder builder);
}
