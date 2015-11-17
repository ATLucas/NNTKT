package containers;

import nlp.Vocab;
import nlp.Word;
import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;
import tools.Logger;

import java.util.ArrayList;

/**
 * Created by Andrew on 11/15/2015.
 */
public class TreeExample {
	private Node root;
	private String labelType;
	private Vocab vocab;

	private ArrayList<Node> leaves;

	public TreeExample(JSONObject jobj, Vocab vocab, String labelType) {
		leaves = new ArrayList<>();
		this.labelType = labelType;
		this.vocab = vocab;
		try {
			root = initNode(jobj);
		} catch(Exception e) {e.printStackTrace();}
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
		if(jobj.has("symbol")) {
			node.word = vocab.id2word(vocab.string2id(jobj.getString("symbol")));
			leaves.add(node);
		} else {
			node.left = initNode(jobj.getJSONObject("left"));
			node.right = initNode(jobj.getJSONObject("right"));
			node.left.parent = node;
			node.right.parent = node;
			node.left.sibling = node.right;
			node.right.sibling = node.left;
		}
		return node;
	}


	private class Node {
		Node left, right, parent, sibling;
		Word word;
		Word targetWord;
		Matrix targetVector;
	}
}
