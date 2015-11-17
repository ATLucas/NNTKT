package tools;

import java.io.IOException;
import java.io.PrintWriter;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Paths;

/**
 * Created by Andrew on 11/11/2015.
 */
public class Common {

	public static String readFileIntoString(String path)
			throws IOException
	{
		byte[] encoded = Files.readAllBytes(Paths.get(path));
		return new String(encoded, Charset.forName("UTF-8"));
	}

	public static void write(String fileName, String data) {
		try {
			PrintWriter writer = new PrintWriter(fileName);
			writer.print(data);
			writer.close();
		} catch(Exception e){e.printStackTrace();}
	}
}
