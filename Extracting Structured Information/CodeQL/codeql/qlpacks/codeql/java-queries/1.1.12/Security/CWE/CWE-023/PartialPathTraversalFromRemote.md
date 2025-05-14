# Partial path traversal vulnerability from remote
A common way to check that a user-supplied path `SUBDIR` falls inside a directory `DIR` is to use `getCanonicalPath()` to remove any path-traversal elements and then check that `DIR` is a prefix. However, if `DIR` is not slash-terminated, this can unexpectedly allow accessing siblings of `DIR`.

See also `java/partial-path-traversal`, which is similar to this query, but may also flag non-remotely-exploitable instances of partial path traversal vulnerabilities.


## Recommendation
If the user should only access items within a certain directory `DIR`, ensure that `DIR` is slash-terminated before checking that `DIR` is a prefix of the user-provided path, `SUBDIR`. Note, Java's `getCanonicalPath()` returns a **non**-slash-terminated path string, so a slash must be added to `DIR` if that method is used.


## Example
In this example, the `if` statement checks if `parent.getCanonicalPath()` is a prefix of `dir.getCanonicalPath()`. However, `parent.getCanonicalPath()` is not slash-terminated. This means that users that supply `dir` may be also allowed to access siblings of `parent` and not just children of `parent`, which is a security issue.


```java
public class PartialPathTraversalBad {
    public void example(File dir, File parent) throws IOException {
        if (!dir.getCanonicalPath().startsWith(parent.getCanonicalPath())) {
            throw new IOException("Path traversal attempt: " + dir.getCanonicalPath());
        }
    }
}

```
In this example, the `if` statement checks if `parent.toPath()` is a prefix of `dir.normalize()`. Because `Path#startsWith` does the correct check that `dir` is a child of `parent`, users will not be able to access siblings of `parent`, as desired.


```java
import java.io.File;

public class PartialPathTraversalGood {
    public void example(File dir, File parent) throws IOException {
        if (!dir.toPath().normalize().startsWith(parent.toPath())) {
            throw new IOException("Path traversal attempt: " + dir.getCanonicalPath());
        }
    }
}

```

## References
* OWASP: [Partial Path Traversal](https://owasp.org/www-community/attacks/Path_Traversal).
* CVE-2022-23457: [ ESAPI Vulnerability Report](https://github.com/ESAPI/esapi-java-legacy/blob/develop/documentation/GHSL-2022-008_The_OWASP_Enterprise_Security_API.md).
* Common Weakness Enumeration: [CWE-23](https://cwe.mitre.org/data/definitions/23.html).
