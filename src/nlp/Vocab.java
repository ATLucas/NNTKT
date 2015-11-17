package nlp;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;
import tools.Logger;

import java.util.HashMap;
import java.util.Random;

/**
 * Created by Andrew on 11/15/2015.
 */
public class Vocab {
	private int wordDim;

	private HashMap<String,Integer> string2id;
	private HashMap<Integer,String> id2string;
	private HashMap<Integer,Word> id2word;

	public Vocab(JSONObject jobj) throws JSONException{
		string2id = new HashMap<>();
		id2string = new HashMap<>();
		id2word = new HashMap<>();

		int size = jobj.getInt("size");
		wordDim = jobj.getInt("wordDim");
		JSONArray words = jobj.getJSONArray("words");
		if(size != words.length()) Logger.die("numWords in vocab file is incorrect");
		Random rand = new Random();
		for(int i=0; i<size; i++) {
			JSONObject wordObj = words.getJSONObject(i);
			String wordString = wordObj.getString("word");

			JSONArray jarray = wordObj.getJSONArray("vector");
			float[] vec = new float[wordDim];
			if(jarray.length() == 0) {
				for(int j=0; j<vec.length; j++) {
					vec[j] = (float)rand.nextGaussian();
				}
			} else if(jarray.length() == wordDim) {
				for(int j=0; j<vec.length; j++) {
					vec[j] = (float)jarray.getDouble(j);
				}
			} else {
				Logger.die("Invalid word vector dimension");
			}

			jarray = wordObj.getJSONArray("matrix");
			float[] mat = new float[wordDim*wordDim];
			if(jarray.length() == 0) {
				for(int j=0; j<mat.length; j++) {
					mat[j] = (float)(rand.nextGaussian()/Math.sqrt(wordDim));
				}
			} else if(jarray.length() == wordDim*wordDim) {
				for(int j=0; j<mat.length; j++) {
					mat[j] = (float)jarray.getDouble(j);
				}
			} else {
				Logger.die("Invalid word matrix dimension");
			}

			string2id.put(wordString, i);
			id2string.put(i, wordString);
			id2word.put(i, new Word(wordDim, i, vec, mat, wordObj.getBoolean("updatable")));
		}
	}

	public int wordDim() {
		return wordDim;
	}

	public int string2id(String s) {
		return string2id.get(s);
	}

	public String id2string(int i) {
		return id2string.get(i);
	}

	public Word id2word(int i) {
		return id2word.get(i);
	}

}
