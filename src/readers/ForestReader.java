package readers;

import containers.Matrix;
import containers.NetworkForest;
import containers.NetworkTree;
import network.NetworkNode;
import nlp.Vocab;
import org.json.JSONArray;
import org.json.JSONObject;
import tools.Common;
import tools.Logger;

/**
 * Created by Andrew on 11/15/2015.
 */
public class ForestReader {

	public static void Read(NetworkForest networkForest, Vocab vocab, String treeFileName,
							NetworkNode networkNode, Matrix weightsCombiner) {
		try {
			JSONObject jobject = new JSONObject(Common.readFileIntoString(treeFileName));
			int numTrees = jobject.getInt("numTrees");
			String labelType = jobject.getString("labelType");
			JSONArray jarray = jobject.getJSONArray("trees");
			if(numTrees != jarray.length()) Logger.die("Invalid number of trees in networkForest file");
			for(int i=0; i<numTrees; i++) {
				networkForest.add(new NetworkTree(jarray.getJSONObject(i), vocab, labelType,
						networkNode, weightsCombiner));
			}
		} catch(Exception e){e.printStackTrace();}
	}
}
