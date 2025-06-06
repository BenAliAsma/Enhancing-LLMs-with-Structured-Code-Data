# Arbitrary file access during archive extraction ("Zip Slip")
Extracting files from a malicious zip file, or similar type of archive, is at risk of directory traversal attacks if filenames from the archive are not properly validated.

Zip archives contain archive entries representing each file in the archive. These entries include a file path for the entry, but these file paths are not restricted and may contain unexpected special elements such as the directory traversal element (`..`). If these file paths are used to create a filesystem path, then a file operation may happen in an unexpected location. This can result in sensitive information being revealed or deleted, or an attacker being able to influence behavior by modifying unexpected files.

For example, if a zip file contains a file entry `..\sneaky-file`, and the zip file is extracted to the directory `c:\output`, then naively combining the paths would result in an output file path of `c:\output\..\sneaky-file`, which would cause the file to be written to `c:\sneaky-file`.


## Recommendation
Ensure that output paths constructed from zip archive entries are validated to prevent writing files to unexpected locations.

The recommended way of writing an output file from a zip archive entry is to verify that the normalized full path of the output file starts with a prefix that matches the destination directory. Path normalization can be done with either `java.io.File.getCanonicalFile()` or `java.nio.file.Path.normalize()`. Prefix checking can be done with `String.startsWith(..)`, but it is better to use `java.nio.file.Path.startsWith(..)`, as the latter works on complete path segments.

Another alternative is to validate archive entries against a whitelist of expected files.


## Example
In this example, a file path taken from a zip archive item entry is combined with a destination directory. The result is used as the destination file path without verifying that the result is within the destination directory. If provided with a zip file containing an archive path like `..\sneaky-file`, then this file would be written outside the destination directory.


```java
void writeZipEntry(ZipEntry entry, File destinationDir) {
    File file = new File(destinationDir, entry.getName());
    FileOutputStream fos = new FileOutputStream(file); // BAD
    // ... write entry to fos ...
}

```
To fix this vulnerability, we need to verify that the normalized `file` still has `destinationDir` as its prefix, and throw an exception if this is not the case.


```java
void writeZipEntry(ZipEntry entry, File destinationDir) {
    File file = new File(destinationDir, entry.getName());
    if (!file.toPath().normalize().startsWith(destinationDir.toPath()))
        throw new Exception("Bad zip entry");
    FileOutputStream fos = new FileOutputStream(file); // OK
    // ... write entry to fos ...
}

```

## References
* Snyk: [Zip Slip Vulnerability](https://snyk.io/research/zip-slip-vulnerability).
* OWASP: [Path Traversal](https://owasp.org/www-community/attacks/Path_Traversal).
* Common Weakness Enumeration: [CWE-22](https://cwe.mitre.org/data/definitions/22.html).
