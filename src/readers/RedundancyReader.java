package readers;

import containers.Dataset;
import containers.Example;
import containers.SentenceDataset;
import edu.berkeley.nlp.lm.ContextEncodedProbBackoffLm;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Created by Andrew on 1/9/2016.
 */
public class RedundancyReader {
	public static Pattern sentencePattern = Pattern.compile(" ([^\\s]+)");
	public static Pattern rangePattern = Pattern.compile("A (\\d+) (\\d+)");

	public static void Read(Dataset dataset, SentenceDataset sentenceDataset, String redFileName,
							ContextEncodedProbBackoffLm<String> lm) {
		ArrayList<String> lines = new ArrayList<>();
		try (BufferedReader br = new BufferedReader(new FileReader(redFileName))) {
			String line;
			while ((line = br.readLine()) != null) {
				lines.add(line);
			}
			br.close();
		} catch (IOException e) {
			e.printStackTrace();
		}

		for(int i=0; i<lines.size(); i+=3) {
			Matcher matcher = rangePattern.matcher(lines.get(i + 1));
			matcher.find();
			int start = Integer.parseInt(matcher.group(1));
			int end = Integer.parseInt(matcher.group(2));
			if(end - start == 1) {
				matcher = sentencePattern.matcher(lines.get(i));

				ArrayList<String> words = new ArrayList<>();
				ArrayList<Float> unigram = new ArrayList<>();
				ArrayList<Float> trigram = new ArrayList<>();
				while (matcher.find()) {
					String word = matcher.group(1);
					words.add(word);
					ArrayList<String> arr = new ArrayList<>();
					arr.add(word);
					unigram.add(lm.getLogProb(arr));
					ArrayList<String> context = new ArrayList<>();
					if(words.size()>2) {
						context.add(words.get(words.size() - 3));
					}
					if(words.size()>1) {
						context.add(words.get(words.size() - 2));
					}
					context.add(word);
					trigram.add(lm.getLogProb(context));
				}

				ArrayList<Float> fluency = new ArrayList<>();
				for(int j=0; j<words.size(); j++) {
					ArrayList<String> del = new ArrayList<>();
					del.addAll(words);
					del.remove(j);
					fluency.add(lm.scoreSentence(del));
				}

				// get pos
				// get previous pos

				ArrayList<Example> sentence = new ArrayList<>();
				for(int j=0; j<words.size(); j++) {

					float red, notRed;
					if(j == start) {
						red = 1f;
						notRed = 0f;
					} else {
						red = 0f;
						notRed = 1f;
					}

//					(float)Math.exp(unigram.get(j))
//					(float)Math.exp(trigram.get(j))

					Example example = new Example(
							new float[]{
									(float)Math.exp(fluency.get(j))
							},
							new float[]{red, notRed});

					dataset.add(example);
					if(red == 1f) {
						for(int k=0; k<20; k++) {
							dataset.add(example);
						}
					}
					sentence.add(example);
				}
				sentenceDataset.add(sentence, words);
			}
		}
	}
}
