package readers;

import containers.MvRnnForest;
import containers.MvRnnTree;
import nlp.Vocab;
import org.json.JSONArray;
import org.json.JSONObject;
import tools.Common;
import tools.Logger;

/**
 * Created by Andrew on 11/15/2015.
 */
public class ForestReader {

	public static void Read(MvRnnForest mvRnnForest, Vocab vocab, String treeFileName) {
		try {
			JSONObject jobject = new JSONObject(Common.readFileIntoString(treeFileName));
			int numTrees = jobject.getInt("numTrees");
			String labelType = jobject.getString("labelType");
			JSONArray jarray = jobject.getJSONArray("trees");
			if(numTrees != jarray.length()) Logger.die("Invalid number of trees in mvRnnForest file");
			for(int i=0; i<numTrees; i++) {
				mvRnnForest.add(new MvRnnTree(jarray.getJSONObject(i), vocab, labelType));
			}
		} catch(Exception e){e.printStackTrace();}
	}
}
