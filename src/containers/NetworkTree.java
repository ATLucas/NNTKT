package containers;

import network.NetworkNode;
import nlp.Vocab;
import nlp.Word;
import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;
import tools.Logger;
import tools.MvRnnTrainConfig;

/**
 * Created by Andrew on 12/12/2015.
 */
public class NetworkTree {
	private TreeNode root;
	private String labelType;
	private Vocab vocab;

	public NetworkTree(JSONObject jobj, Vocab vocab, String labelType,
					   NetworkNode networkNode, Matrix weightsCombiner) {
		this.labelType = labelType;
		this.vocab = vocab;
		try {
			root = initNode(jobj, networkNode, weightsCombiner);
		} catch(Exception e) {e.printStackTrace();}
	}

	private TreeNode initNode(JSONObject jobj, NetworkNode networkNode, Matrix weightsCombiner) throws JSONException {
		TreeNode node = new TreeNode();
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
			if(!jobj.has("left") || !jobj.has("right")) {
				Logger.die("Tried to initialize a TreeNode without a word " +
						"and missing a left or right child");
			}
			node.networkNode = new NetworkNode(networkNode);
			node.weightsCombiner = weightsCombiner;
			node.left = initNode(jobj.getJSONObject("left"), networkNode, weightsCombiner);
			node.right = initNode(jobj.getJSONObject("right"), networkNode, weightsCombiner);
		}
		return node;
	}

	public MvPair getTopLabel() {
		return root.targetWord.mvPair;
	}

	public float train(MvRnnTrainConfig config) {
		root.forward();
		float cost = root.backward(config, null);
		root.update(config);
		return cost;
	}

	public MvPair decode() {
		return root.forward();
	}

	private class TreeNode {

		TreeNode left;
		TreeNode right;

		NetworkNode networkNode;
		Matrix weightsCombiner;
		Word word;

		Word targetWord;
		Matrix targetVector;

		MvPair input, output, error;

		public MvPair forward() {
			if(word != null) {
				return output = word.mvPair;
			} else {
				MvPair leftPair = left.forward();
				MvPair rightPair = right.forward();

				Matrix leftVector = leftPair.vector.multiply(rightPair.matrix);
				Matrix rightVector = rightPair.vector.multiply(leftPair.matrix);

				input = new MvPair(Matrix.StackHorizontally(leftVector, rightVector),
											Matrix.StackHorizontally(leftPair.matrix, rightPair.matrix));
				output = new MvPair(networkNode.forward(input.vector),
											input.matrix.multiply(weightsCombiner));
				return output;
			}
		}

		public float backward(MvRnnTrainConfig config, MvPair error) {
			float cost = 0;

			if(targetWord != null) {
				error = new MvPair(output);
				cost += networkNode.calcObjectiveFunction(config.networkConfig, error.vector, targetWord.mvPair.vector);
				cost += error.matrix.applyMeanSquaredError(targetWord.mvPair.matrix, 1);
			} else if(targetVector != null) {
				Logger.die("Target vectors not yet supported for NetworkTree");
			}

			this.error = new MvPair(error);

			if(networkNode != null) {
				error.vector = networkNode.backward(error.vector);
				error.matrix = error.matrix.multiplyTranspose(weightsCombiner);

				if (left != null) {
					if (right != null) {
						cost += left.backward(
								config, new MvPair(
										error.vector.getLeftHalf().multiplyTranspose(right.output.matrix),
										error.matrix.getLeftHalf()));
						cost += right.backward(
								config, new MvPair(
										error.vector.getRightHalf().multiplyTranspose(left.output.matrix),
										error.matrix.getRightHalf()));
					} else {
						Logger.die("Hit an invalid node");
						return 0;
					}
				}
			}

			return cost;
		}

		public void update(MvRnnTrainConfig config) {
			if(word != null) {
				word.update(error.vector, error.matrix,
						config.vectorLearningRate,
						config.matrixLearningRate);
			} else {
				networkNode.update(config.networkConfig);
				weightsCombiner.updateWeights(input.matrix.transposeMultiply(error.matrix),
						config.weightsLearningRate,
						config.weightsDecayFactor);
				left.update(config);
				right.update(config);
			}
		}
	}
}
