package readers;

import containers.Forest;
import containers.TreeExample;
import nlp.Vocab;
import org.json.JSONArray;
import org.json.JSONObject;
import tools.Common;
import tools.Logger;

/**
 * Created by Andrew on 11/15/2015.
 */
public class ForestReader {

	public static void Read(Forest forest, Vocab vocab, String treeFileName) {
		try {
			JSONObject jobject = new JSONObject(Common.readFileIntoString(treeFileName));
			int numTrees = jobject.getInt("numTrees");
			String labelType = jobject.getString("labelType");
			JSONArray jarray = jobject.getJSONArray("trees");
			if(numTrees != jarray.length()) Logger.die("Invalid number of trees in forest file");
			for(int i=0; i<numTrees; i++) {
				forest.add(new TreeExample(jarray.getJSONObject(i), vocab, labelType));
			}
		} catch(Exception e){e.printStackTrace();}
	}
}
