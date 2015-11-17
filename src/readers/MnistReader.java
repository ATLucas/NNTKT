package readers;

import containers.Dataset;
import containers.Example;
import tools.Logger;

import java.io.File;
import java.io.FileInputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * Created by Andrew on 11/11/2015.
 */
public class MnistReader {

	public static void Read(Dataset dataset, String imageFileName, String labelFileName) {
		try {
			File labelFile = new File(labelFileName);
			FileInputStream labelsIn = new FileInputStream(labelFile);
			byte[] labelArray = new byte[(int) labelFile.length()];
			labelsIn.read(labelArray);
			ByteBuffer labels = ByteBuffer.wrap(labelArray);
			labels.order(ByteOrder.BIG_ENDIAN);

			File imageFile = new File(imageFileName);
			FileInputStream imagesIn = new FileInputStream(imageFile);
			byte[] imageArray = new byte[(int) imageFile.length()];
			imagesIn.read(imageArray);
			ByteBuffer images = ByteBuffer.wrap(imageArray);
			images.order(ByteOrder.BIG_ENDIAN);

			int labelsMagicNumber = labels.getInt();
			int imagesMagicNumber = images.getInt();
			if(labelsMagicNumber == 2049 && imagesMagicNumber == 2051) {
				int numLabels = labels.getInt();
				int numImages = images.getInt();
				if(numLabels != numImages) Logger.die("Could not read mnist file");

				int numRows = images.getInt();
				int numCols = images.getInt();
				for (int i = 0; i < numLabels; i++) {
					float[] lbls = new float[10];
					lbls[labels.get() & 0xFF] = 1f;

					float[] imgs = new float[numRows * numCols];
					for(int j=0; j<imgs.length; j++) {
						imgs[j] = (float)(images.get() & 0xFF) / 255f;
					}

					dataset.add(new Example(imgs, lbls));
				}
			} else {
				Logger.die("Could not read mnist file");
			}
		} catch (Exception e) {e.printStackTrace();}
	}
}
