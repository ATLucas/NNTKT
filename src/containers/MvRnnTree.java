package containers;

import network.NeuralNetwork;
import nlp.Vocab;
import nlp.Word;
import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;
import tools.Logger;
import tools.TrainConfig;

import java.util.ArrayList;
import java.util.Stack;

/**
 * Created by Andrew on 11/15/2015.
 */
public class MvRnnTree {
	private Node root;
	private String labelType;
	private Vocab vocab;

	private ArrayList<Node> leaves;

	public MvRnnTree(JSONObject jobj, Vocab vocab, String labelType) {
		leaves = new ArrayList<>();
		this.labelType = labelType;
		this.vocab = vocab;
		try {
			root = initNode(jobj);
		} catch(Exception e) {e.printStackTrace();}
	}

	public MvPair getTopLabel() {
		return root.targetWord.mvPair;
	}

	private Node initNode(JSONObject jobj) throws JSONException {
		Node node = new Node();
		if(jobj.has("label")){
			if(labelType.equals("String")) {
				node.targetWord = vocab.id2word(vocab.string2id(jobj.getString("label")));
			} else if(labelType.equals("vector")) {
				JSONArray jarray = jobj.getJSONArray("label");
				float[] vec = new float[jarray.length()];
				for(int i=0; i<vec.length; i++) {
					vec[i] = (float)jarray.getDouble(i);
				}
				node.targetVector = new Matrix(1, vocab.wordDim());
				node.targetVector.copyRow(vec);
			} else Logger.die("Unsupported label type");
		}
		if(jobj.has("word")) {
			node.word = vocab.id2word(vocab.string2id(jobj.getString("word")));
		} else {
			node.left = initNode(jobj.getJSONObject("left"));
			node.right = initNode(jobj.getJSONObject("right"));
		}
		return node;
	}

	public float train(TrainConfig config, NeuralNetwork network, Matrix weights) {
		return root.backprop(config, network, weights, root.forward(network, weights));
	}

	public MvPair decode(NeuralNetwork network, Matrix weights) {
		return root.forward(network, weights);
	}

	private static MvPair Collapse(NeuralNetwork network, Matrix weights, MvPair left, MvPair right) {
		Matrix firstVector = left.vector.multiply(right.matrix);
		Matrix secondVector = right.vector.multiply(left.matrix);
		Matrix weightsInput = Matrix.StackHorizontally(left.matrix, right.matrix);
		MvPair mvPair = new MvPair(network.forward(Matrix.StackHorizontally(firstVector, secondVector)),
							weightsInput.multiply(weights));
		mvPair.input = weightsInput;
		return mvPair;
	}

	private class Node {
		Node left, right;
		Word word;
		Word targetWord;
		Matrix targetVector;
		MvPair mvPair;

		Stack<Matrix> inputs;

		Node(){
			inputs = new Stack<>();
		}

		MvPair forward(NeuralNetwork network, Matrix weights) {
			if(left != null && right != null) {
				MvPair l = left.forward(network, weights);
				MvPair r = right.forward(network, weights);
				mvPair = Collapse(network, weights, l, r);
				inputs.push(mvPair.input);
				return mvPair;
			} else {
				if(word != null) {
					return mvPair = word.mvPair;
				} else {
					Logger.die("Hit an invalid node");
					return null;
				}
			}
		}

		float backprop(TrainConfig config, NeuralNetwork network, Matrix weights, MvPair data) {
			float cost = 0;

			if(targetWord != null) {
				cost = network.calcObjectiveFunction(config, data.vector, targetWord.mvPair.vector);
				data.matrix.applyMeanSquaredError(config, targetWord.mvPair.matrix);
			} else if(targetVector != null) {
				Logger.die("Target vectors not yet supported for tree");
			}

			if(word != null) {
				word.update(data.vector, data.matrix);
			} else {
				data.vector = network.backward(data.vector);
				network.update(config);
				if(inputs.size() == 0) Logger.die("Tried to backprop on a node that has not received input");
				Matrix temp = data.matrix.multiplyTranspose(weights);
				//todo weights should use its own learning rate and decay factor
				weights.updateWeights(inputs.pop().transposeMultiply(data.matrix), config.learningRate, config.decayFactor);
				data.matrix = temp;

				if (left != null) {
					if (right != null) {
						Matrix leftError = data.vector.getLeftHalf().multiplyTranspose(right.mvPair.matrix);
						Matrix rightError = data.vector.getRightHalf().multiplyTranspose(left.mvPair.matrix);
						cost += right.backprop(config, network, weights,
								new MvPair(rightError, data.matrix.getRightHalf()));
						cost += left.backprop(config, network, weights,
								new MvPair(leftError, data.matrix.getLeftHalf()));
					} else {
						Logger.die("Hit an invalid node");
						return 0;
					}
				}
			}

			return cost;
		}
	}


}
