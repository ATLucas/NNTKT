package tools;

import java.io.PrintWriter;
import java.text.SimpleDateFormat;
import java.util.Calendar;

/**
 * Created by Andrew on 11/11/2015.
 */
public class Logger {

	private static PrintWriter LOG;

	private static final SimpleDateFormat sdf = new SimpleDateFormat("HH:mm:ss");
	private static final SimpleDateFormat sdf2 = new SimpleDateFormat("HH-mm-ss");

	public static String timeString;

	public static void init(String logDirectory) {
		timeString = sdf2.format(Calendar.getInstance().getTime());
		try {
			LOG = new PrintWriter(logDirectory+"/"+timeString+".log");
		} catch(Exception e) {
			e.printStackTrace();
		}
	}

	public static String time() {
		return sdf.format(Calendar.getInstance().getTime());
	}

	public static void log(String message, boolean addNewline) {
		if(LOG != null) LOG.print(message);
		System.out.print(message);
		if(addNewline) {
			if(LOG != null) LOG.println();
			System.out.println();
		}
	}

	public static void log(String message) {
		log(message, true);
	}

	public static void warn(String message) {
		message = "WARNING: " + message;
		if(LOG != null) LOG.println(message);
		System.err.println(message);
	}

	public static void die(String message) {
		message = "ERROR: " + message;
		if(LOG != null) LOG.println(message);
		System.err.println(message);
		close();
		System.exit(-1);
	}

	public static void flush() {
		if(LOG != null) LOG.flush();
	}

	public static void close() {
		if(LOG != null) LOG.close();
	}
}
