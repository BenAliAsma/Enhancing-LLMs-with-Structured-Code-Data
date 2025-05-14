# Local information disclosure in a temporary directory
Local information disclosure can occur when files/directories are written into directories that are shared between all users on the system.

On most [unix-like](https://en.wikipedia.org/wiki/Unix-like) systems, the system temporary directory is shared between local users. If files/directories are created within the system temporary directory without using APIs that explicitly set the correct file permissions, local information disclosure can occur.

Depending upon the particular file contents exposed, this vulnerability can have a [CVSSv3.1 base score of 6.2/10](https://nvd.nist.gov/vuln-metrics/cvss/v3-calculator?vector=AV:L/AC:L/PR:N/UI:N/S:U/C:H/I:N/A:N&version=3.1).


## Recommendation
Use JDK methods that specifically protect against this vulnerability:

* [java.nio.file.Files.createTempDirectory](https://docs.oracle.com/javase/8/docs/api/java/nio/file/Files.html#createTempDirectory-java.nio.file.Path-java.lang.String-java.nio.file.attribute.FileAttribute...-)
* [java.nio.file.Files.createTempFile](https://docs.oracle.com/javase/8/docs/api/java/nio/file/Files.html#createTempFile-java.nio.file.Path-java.lang.String-java.lang.String-java.nio.file.attribute.FileAttribute...-)
Otherwise, create the file/directory by manually specifying the expected posix file permissions. For example: `PosixFilePermissions.asFileAttribute(EnumSet.of(PosixFilePermission.OWNER_READ, PosixFilePermission.OWNER_WRITE))`

* [java.nio.file.Files.createFile](https://docs.oracle.com/javase/8/docs/api/java/nio/file/Files.html#createFile-java.nio.file.Path-java.nio.file.attribute.FileAttribute...-)
* [java.nio.file.Files.createDirectory](https://docs.oracle.com/javase/8/docs/api/java/nio/file/Files.html#createDirectory-java.nio.file.Path-java.nio.file.attribute.FileAttribute...-)
* [java.nio.file.Files.createDirectories](https://docs.oracle.com/javase/8/docs/api/java/nio/file/Files.html#createDirectories-java.nio.file.Path-java.nio.file.attribute.FileAttribute...-)

## Example
In the following example, files and directories are created with file permissions that allow other local users to read their contents.


```java
import java.io.File;

public class TempDirUsageVulnerable {
    void exampleVulnerable() {
        File temp1 = File.createTempFile("random", ".txt"); // BAD: File has permissions `-rw-r--r--`

        File temp2 = File.createTempFile("random", "file", null); // BAD: File has permissions `-rw-r--r--`

        File systemTempDir = new File(System.getProperty("java.io.tmpdir"));
        File temp3 = File.createTempFile("random", "file", systemTempDir); // BAD: File has permissions `-rw-r--r--`

        File tempDir = com.google.common.io.Files.createTempDir(); // BAD: CVE-2020-8908: Directory has permissions `drwxr-xr-x`

        new File(System.getProperty("java.io.tmpdir"), "/child").mkdir(); // BAD: Directory has permissions `-rw-r--r--`

        File tempDirChildFile = new File(System.getProperty("java.io.tmpdir"), "/child-create-file.txt");
        Files.createFile(tempDirChildFile.toPath()); // BAD: File has permissions `-rw-r--r--`

        File tempDirChildDir = new File(System.getProperty("java.io.tmpdir"), "/child-dir");
        tempDirChildDir.mkdir(); // BAD: Directory has permissions `drwxr-xr-x`
        Files.createDirectory(tempDirChildDir.toPath()); // BAD: Directory has permissions `drwxr-xr-x`
    }
}

```
In the following example, files and directories are created with file permissions that protect their contents.


```java
import java.io.File;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.attribute.PosixFilePermission;
import java.nio.file.attribute.PosixFilePermissions;

import java.util.EnumSet;


public class TempDirUsageSafe {
    void exampleSafe() throws IOException {
        Path temp1 = Files.createTempFile("random", ".txt"); // GOOD: File has permissions `-rw-------`

        Path temp2 = Files.createTempDirectory("random-directory"); // GOOD: File has permissions `drwx------`

        // Creating a temporary file with a non-randomly generated name
        File tempChildFile = new File(System.getProperty("java.io.tmpdir"), "/child-create-file.txt");
        // Warning: This will fail on windows as it doesn't support PosixFilePermissions.
        // See `exampleSafeWithWindowsSupportFile` if your code needs to support windows and unix-like systems.
        Files.createFile(
            tempChildFile.toPath(),
            PosixFilePermissions.asFileAttribute(EnumSet.of(PosixFilePermission.OWNER_READ, PosixFilePermission.OWNER_WRITE))
        ); // GOOD: Good has permissions `-rw-------`
    }

    /*
     * An example of a safe use of createFile or createDirectory if your code must support windows and unix-like systems.
     */
    void exampleSafeWithWindowsSupportFile() {
        // Creating a temporary file with a non-randomly generated name
        File tempChildFile = new File(System.getProperty("java.io.tmpdir"), "/child-create-file.txt");
        createTempFile(tempChildFile.toPath()); // GOOD: Good has permissions `-rw-------`
    }

    static void createTempFile(Path tempDirChild) {
        try {
            if (tempDirChild.getFileSystem().supportedFileAttributeViews().contains("posix")) {
                // Explicit permissions setting is only required on unix-like systems because
                // the temporary directory is shared between all users.
                // This is not necessary on Windows, each user has their own temp directory
                final EnumSet<PosixFilePermission> posixFilePermissions =
                        EnumSet.of(
                            PosixFilePermission.OWNER_READ,
                            PosixFilePermission.OWNER_WRITE
                        );
                if (!Files.exists(tempDirChild)) {
                    Files.createFile(
                        tempDirChild,
                        PosixFilePermissions.asFileAttribute(posixFilePermissions)
                    ); // GOOD: Directory has permissions `-rw-------`
                } else {
                    Files.setPosixFilePermissions(
                            tempDirChild,
                            posixFilePermissions
                    ); // GOOD: Good has permissions `-rw-------`, or will throw an exception if this fails
                }
            } else if (!Files.exists(tempDirChild)) {
                // On Windows, we still need to create the directory, when it doesn't already exist.
                Files.createDirectory(tempDirChild); // GOOD: Windows doesn't share the temp directory between users
            }
        } catch (IOException exception) {
            throw new UncheckedIOException("Failed to create temp file", exception);
        }
    }

    void exampleSafeWithWindowsSupportDirectory() {
        File tempDirChildDir = new File(System.getProperty("java.io.tmpdir"), "/child-dir");
        createTempDirectories(tempDirChildDir.toPath()); // GOOD: Directory has permissions `drwx------`
    }

    static void createTempDirectories(Path tempDirChild) {
        try {
            if (tempDirChild.getFileSystem().supportedFileAttributeViews().contains("posix")) {
                // Explicit permissions setting is only required on unix-like systems because
                // the temporary directory is shared between all users.
                // This is not necessary on Windows, each user has their own temp directory
                final EnumSet<PosixFilePermission> posixFilePermissions =
                        EnumSet.of(
                            PosixFilePermission.OWNER_READ,
                            PosixFilePermission.OWNER_WRITE,
                            PosixFilePermission.OWNER_EXECUTE
                        );
                if (!Files.exists(tempDirChild)) {
                    Files.createDirectories(
                        tempDirChild,
                        PosixFilePermissions.asFileAttribute(posixFilePermissions)
                    ); // GOOD: Directory has permissions `drwx------`
                } else {
                    Files.setPosixFilePermissions(
                            tempDirChild,
                            posixFilePermissions
                    ); // GOOD: Good has permissions `drwx------`, or will throw an exception if this fails
                }
            } else if (!Files.exists(tempDirChild)) {
                // On Windows, we still need to create the directory, when it doesn't already exist.
                Files.createDirectories(tempDirChild); // GOOD: Windows doesn't share the temp directory between users
            }
        } catch (IOException exception) {
            throw new UncheckedIOException("Failed to create temp dir", exception);
        }
    }
}

```

## References
* OWASP: [Insecure Temporary File](https://owasp.org/www-community/vulnerabilities/Insecure_Temporary_File).
* CERT: [FIO00-J. Do not operate on files in shared directories](https://wiki.sei.cmu.edu/confluence/display/java/FIO00-J.+Do+not+operate+on+files+in+shared+directories).
* Common Weakness Enumeration: [CWE-200](https://cwe.mitre.org/data/definitions/200.html).
* Common Weakness Enumeration: [CWE-732](https://cwe.mitre.org/data/definitions/732.html).
