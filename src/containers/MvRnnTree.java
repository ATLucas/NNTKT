package containers;

import network.NeuralNetwork;
import nlp.Vocab;
import nlp.Word;
import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;
import tools.Logger;
import tools.MvRnnTrainConfig;

import java.util.ArrayList;

/**
 * Created by Andrew on 11/15/2015.
 */
public class MvRnnTree {
	private Node root;
	private String labelType;
	private Vocab vocab;

	public MvRnnTree(JSONObject jobj, Vocab vocab, String labelType) {
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

	public float train(MvRnnTrainConfig config, NeuralNetwork network, Matrix weights) {
		return root.backprop(config, network, weights, null); // feed-forward is done at every node in the tree
	}

	public MvPair decode(NeuralNetwork network, Matrix weights) {
		return root.forward(network, weights);
	}

	private class Node {
		Node left, right;
		Word word;
		Word targetWord;
		Matrix targetVector;
		MvPair input, output;

		MvPair forward(NeuralNetwork network, Matrix weights) {
			if(left != null && right != null) {
				MvPair l = left.forward(network, weights);
				MvPair r = right.forward(network, weights);

				Matrix firstVector = l.vector.multiply(r.matrix);
				Matrix secondVector = r.vector.multiply(l.matrix);

				input = new MvPair(Matrix.StackHorizontally(firstVector, secondVector),
						Matrix.StackHorizontally(l.matrix, r.matrix));
				output = new MvPair(network.forward(input.vector),
						input.matrix.multiply(weights));
				return output;
			} else {
				if(word != null) {
					output = word.mvPair;
					return output;
				} else {
					Logger.die("Hit an invalid node");
					return null;
				}
			}
		}

		float backprop(MvRnnTrainConfig config, NeuralNetwork network, Matrix weights, MvPair error) {
			float cost = 0;
			this.forward(network, weights);

			/** If we have a label, calculate the error,
			 * otherwise use the back-propped error **/
			if(targetWord != null) {
				error = new MvPair(output);
				cost += network.calcObjectiveFunction(config.networkConfig, error.vector, targetWord.mvPair.vector);
				cost += error.matrix.applyMeanSquaredError(targetWord.mvPair.matrix, 1);
			} else if(targetVector != null) {
				Logger.die("Target vectors not yet supported for MvRnnTree");
			}

			if(word != null) { // if this node is a leaf
				word.update(error.vector, error.matrix, config.vectorLearningRate, config.matrixLearningRate);
			} else {
				/** Backprop the error **/
				error.vector = network.backward(error.vector);
				Matrix temp = error.matrix.multiplyTranspose(weights);

				/** Update the network and weights **/
				if(input == null) Logger.die("Tried to backprop on a node that has not received input");
				network.update(config.networkConfig);
				weights.updateWeights(input.matrix.transposeMultiply(error.matrix),
						config.weightsLearningRate,
						config.weightsDecayFactor);
				error.matrix = temp;

				/** Now continue the backprop **/
				if (left != null) {
					if (right != null) {
						cost += right.backprop(config, network, weights,
										new MvPair(
											error.vector.getRightHalf().multiplyTranspose(left.output.matrix),
											error.matrix.getRightHalf()));
						cost += left.backprop(config, network, weights,
										new MvPair(
											error.vector.getLeftHalf().multiplyTranspose(right.output.matrix),
											error.matrix.getLeftHalf()));
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
