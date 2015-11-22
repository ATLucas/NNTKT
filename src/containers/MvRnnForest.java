package containers;

import java.util.ArrayList;
import java.util.Random;

/**
 * Created by Andrew on 11/15/2015.
 */
public class MvRnnForest {

	private ArrayList<MvRnnTree> trees;
	private Random rand;

	public MvRnnForest() {
		trees = new ArrayList<>();
		rand = new Random();
	}

	public int size() {
		return trees.size();
	}

	public MvRnnTree getTree(int i) {
		if(i<size()) return trees.get(i);
		else return null;
	}

	public void add(MvRnnTree tree) {
		trees.add(tree);
	}

	public MvRnnTree getTree() {
		return trees.get(rand.nextInt(trees.size()));
	}
}
