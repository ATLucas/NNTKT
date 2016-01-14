package containers;

import java.util.ArrayList;

/**
 * Created by Andrew on 1/9/2016.
 */
public class SentenceDataset {
	private ArrayList<ArrayList<Example>> sentenceExamples;
	private ArrayList<ArrayList<String>> sentenceStrings;

	public SentenceDataset() {
		sentenceExamples = new ArrayList<>();
		sentenceStrings = new ArrayList<>();
	}

	public int size() {
		return sentenceExamples.size();
	}

	public void add(ArrayList<Example> sentenceExample, ArrayList<String> sentenceString) {
		sentenceExamples.add(sentenceExample);
		sentenceStrings.add(sentenceString);
	}

	public ArrayList<Example> getSentenceExample(int i) {
		return sentenceExamples.get(i);
	}

	public ArrayList<String> getSentenceString(int i) {
		return sentenceStrings.get(i);
	}
}
