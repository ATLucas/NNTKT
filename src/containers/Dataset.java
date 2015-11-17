package containers;

import java.util.ArrayList;
import java.util.Random;

/**
 * Created by Andrew on 11/11/2015.
 */
public class Dataset {
	private ArrayList<Example> examples;
	private Random rand;

	public Dataset() {
		examples = new ArrayList<>();
		rand = new Random();
	}

	public int size() {
		return examples.size();
	}

	public Example getExample(int i) {
		if(i<size()) return examples.get(i);
		else return null;
	}

	public void add(Example example) {
		examples.add(example);
	}

	public Example getExample() {
		return examples.get(rand.nextInt(examples.size()));
	}

	public Minibatch getMinibatch(int size) {
		Minibatch minibatch = new Minibatch();
		for(int i=0; i<size; i++) { minibatch.add(getExample()); }
		return minibatch;
	}
}
