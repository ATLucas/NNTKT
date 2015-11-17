package network;

import containers.Matrix;
import tools.TrainConfig;

/**
 * Created by Andrew on 11/12/2015.
 */
public interface NetworkComponent {

	int inputDim();
	int outputDim();
	Matrix forward(Matrix matrix);
	Matrix backward(Matrix matrix);
	void update(TrainConfig config);
	void toString(StringBuilder builder);
}
