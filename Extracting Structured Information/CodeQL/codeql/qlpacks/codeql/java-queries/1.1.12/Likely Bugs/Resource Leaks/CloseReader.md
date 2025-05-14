# Potential input resource leak
A subclass of `Reader` or `InputStream` that is opened for reading but not closed may cause a resource leak.


## Recommendation
Ensure that the resource is always closed to avoid a resource leak. Note that, because of exceptions, it is safest to close a resource in a `finally` block. (However, this is unnecessary for subclasses of `CharArrayReader`, `StringReader` and `ByteArrayInputStream`.)

For Java 7 or later, the recommended way to close resources that implement `java.lang.AutoCloseable` is to declare them within a `try-with-resources` statement, so that they are closed implicitly.


## Example
In the following example, the resource `br` is opened but not closed.


```java
public class CloseReader {
	public static void main(String[] args) throws IOException {
		BufferedReader br = new BufferedReader(new FileReader("C:\\test.txt"));
		System.out.println(br.readLine());
		// ...
	}
}
```
In the following example, the resource `br` is opened in a `try` block and later closed in a `finally` block.


```java
public class CloseReaderFix {
	public static void main(String[] args) throws IOException {
		BufferedReader br = null;
		try {
			br = new BufferedReader(new FileReader("C:\\test.txt"));
			System.out.println(br.readLine());
		}
		finally {
			if(br != null)
				br.close();  // 'br' is closed
		}
		// ...
	}
}
```
Note that nested class instance creation expressions of `Reader`s or `InputStream`s are not safe to use if the constructor of the outer expression may throw an exception. In the following example, the `InputStreamReader` may throw an exception, in which case the inner `FileInputStream` is not closed.


```java
public class CloseReaderNested {
	public static void main(String[] args) throws IOException {
		InputStreamReader reader = null;
		try {
			// InputStreamReader may throw an exception, in which case the ...
			reader = new InputStreamReader(
					// ... FileInputStream is not closed by the finally block
					new FileInputStream("C:\\test.txt"), "UTF-8");
			System.out.println(reader.read());
		}
		finally {
			if (reader != null)
				reader.close();
		}
	}
}
```
In this case, the inner expression needs to be assigned to a local variable and closed separately, as shown below.


```java
public class CloseReaderNestedFix {
	public static void main(String[] args) throws IOException {
		FileInputStream fis = null;
		InputStreamReader reader = null;
		try {
			fis = new FileInputStream("C:\\test.txt");
			reader = new InputStreamReader(fis);
			System.out.println(reader.read());
		}
		finally {
			if (reader != null)
				reader.close();  // 'reader' is closed
			if (fis != null)
				fis.close();  // 'fis' is closed
		}
	}
}
```

## References
* IBM developerWorks: [Java theory and practice: Good housekeeping practices](https://web.archive.org/web/20201109041839/http://www.ibm.com/developerworks/java/library/j-jtp03216/index.html).
* The Java Tutorials: [The try-with-resources Statement](https://docs.oracle.com/javase/tutorial/essential/exceptions/tryResourceClose.html).
* Common Weakness Enumeration: [CWE-404](https://cwe.mitre.org/data/definitions/404.html).
* Common Weakness Enumeration: [CWE-772](https://cwe.mitre.org/data/definitions/772.html).
