# Arbitrary file access during archive extraction ("Zip Slip")
Extracting files from a malicious zip file, or similar type of archive, is at risk of directory traversal attacks if filenames from the archive are not properly validated.

Zip archives contain archive entries representing each file in the archive. These entries include a file path for the entry, but these file paths are not restricted and may contain unexpected special elements such as the directory traversal element (`..`). If these file paths are used to create a filesystem path, then a file operation may happen in an unexpected location. This can result in sensitive information being revealed or deleted, or an attacker being able to influence behavior by modifying unexpected files.

For example, if a zip file contains a file entry `..\sneaky-file`, and the zip file is extracted to the directory `c:\output`, then naively combining the paths would result in an output file path of `c:\output\..\sneaky-file`, which would cause the file to be written to `c:\sneaky-file`.


## Recommendation
Ensure that output paths constructed from zip archive entries are validated to prevent writing files to unexpected locations.

The recommended way of writing an output file from a zip archive entry is to:

1. Use `Path.Combine(destinationDirectory, archiveEntry.FullName)` to determine the raw output path.
1. Use `Path.GetFullPath(..)` on the raw output path to resolve any directory traversal elements.
1. Use `Path.GetFullPath(destinationDirectory + Path.DirectorySeparatorChar)` to determine the fully resolved path of the destination directory.
1. Validate that the resolved output path `StartsWith` the resolved destination directory, aborting if this is not true.
Another alternative is to validate archive entries against a whitelist of expected files.


## Example
In this example, a file path taken from a zip archive item entry is combined with a destination directory. The result is used as the destination file path without verifying that the result is within the destination directory. If provided with a zip file containing an archive path like `..\sneaky-file`, then this file would be written outside the destination directory.


```csharp
using System.IO;
using System.IO.Compression;

class Bad
{
    public static void WriteToDirectory(ZipArchiveEntry entry,
                                        string destDirectory)
    {
        string destFileName = Path.Combine(destDirectory, entry.FullName);
        entry.ExtractToFile(destFileName);
    }
}

```
To fix this vulnerability, we need to make three changes. Firstly, we need to resolve any directory traversal or other special characters in the path by using `Path.GetFullPath`. Secondly, we need to identify the destination output directory, again using `Path.GetFullPath`, this time on the output directory. Finally, we need to ensure that the resolved output starts with the resolved destination directory, and throw an exception if this is not the case.


```csharp
using System.IO;
using System.IO.Compression;

class Good
{
    public static void WriteToDirectory(ZipArchiveEntry entry,
                                        string destDirectory)
    {
        string destFileName = Path.GetFullPath(Path.Combine(destDirectory, entry.FullName));
        string fullDestDirPath = Path.GetFullPath(destDirectory + Path.DirectorySeparatorChar);
        if (!destFileName.StartsWith(fullDestDirPath)) {
            throw new System.InvalidOperationException("Entry is outside the target dir: " +
                                                                                 destFileName);
        }
        entry.ExtractToFile(destFileName);
    }
}

```

## References
* Snyk: [Zip Slip Vulnerability](https://snyk.io/research/zip-slip-vulnerability).
* OWASP: [Path Traversal](https://owasp.org/www-community/attacks/Path_Traversal).
* Common Weakness Enumeration: [CWE-22](https://cwe.mitre.org/data/definitions/22.html).
