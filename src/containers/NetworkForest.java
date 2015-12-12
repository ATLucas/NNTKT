package containers;

import java.util.ArrayList;
import java.util.Random;

/**
 * Created by Andrew on 11/15/2015.
 */
public class NetworkForest {

	private ArrayList<NetworkTree> trees;
	private Random rand;

	public NetworkForest() {
		trees = new ArrayList<>();
		rand = new Random();
	}

	public int size() {
		return trees.size();
	}

	public NetworkTree getTree(int i) {
		if(i<size()) return trees.get(i);
		else return null;
	}

	public void add(NetworkTree tree) {
		trees.add(tree);
	}

	public NetworkTree getTree() {
		return trees.get(rand.nextInt(trees.size()));
	}
}
