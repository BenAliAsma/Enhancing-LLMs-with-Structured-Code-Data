# Creates empty ZIP file entry
The `ZipOutputStream` class is used to write ZIP files to a file or other stream. A ZIP file consists of a number of *entries*. Usually each entry corresponds to a file in the directory structure being zipped. There is a method on `ZipOutputStream` that is slightly confusingly named `putNextEntry`. Despite its name, it does not write a whole entry. Instead, it writes the *metadata* for an entry. The content for that entry is then written using the `write` method. Finally the entry is closed using `closeEntry`.

Therefore, if you call `putNextEntry` and `closeEntry` but omit the call to `write`, an empty ZIP file entry is written to the output stream.


## Recommendation
Ensure that you include a call to `ZipOutputStream.write`.


## Example
In the following example, the `archive` method calls `putNextEntry` and `closeEntry` but the call to `write` is left out.


```java
class Archive implements Closeable
{
	private ZipOutputStream zipStream;

	public Archive(File zip) throws IOException {
		OutputStream stream = new FileOutputStream(zip);
		stream = new BufferedOutputStream(stream);
		zipStream = new ZipOutputStream(stream);
	}

	public void archive(String name, byte[] content) throws IOException {
		ZipEntry entry = new ZipEntry(name);
		zipStream.putNextEntry(entry);
		// Missing call to 'write'
		zipStream.closeEntry();
	}

	public void close() throws IOException {
		zipStream.close();
	}
}
```

## References
* Java API Specification: [ ZipOutputStream](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/java/util/zip/ZipOutputStream.html).
