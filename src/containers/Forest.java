package containers;

import java.util.ArrayList;
import java.util.Random;

/**
 * Created by Andrew on 11/15/2015.
 */
public class Forest {

	private ArrayList<TreeExample> trees;
	private Random rand;

	public Forest() {
		trees = new ArrayList<>();
		rand = new Random();
	}

	public int size() {
		return trees.size();
	}

	public TreeExample getTree(int i) {
		if(i<size()) return trees.get(i);
		else return null;
	}

	public void add(TreeExample treeExample) {
		trees.add(treeExample);
	}

	public TreeExample getTree() {
		return trees.get(rand.nextInt(trees.size()));
	}
}
