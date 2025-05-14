# Container size compared to zero
A map, collection, string or array will always have size of at least zero. Checking that an object of one of these types has size greater than or equal to zero will always be true, while checking that it has size less than zero will always be false.


## Recommendation
For collections, maps and strings, if the intention was to check whether the object was empty, is it preferred to use the `isEmpty()` method. For arrays, check that the `length` field is greater than (**not** equal to) zero.


## Example
The following example shows creation of a file guarded by comparison of a string length with zero. This can result in the attempted creation of a file with an empty name.


```java
import java.io.File;

class ContainerSizeCmpZero
{
    private static File MakeFile(String filename) {
    if(filename != null && filename.length() >= 0) {
        return new File(filename);
    }
    return new File("default.name");
    }
}

```
In the following revised example, the check against zero has been replaced with a call to `isEmpty()`. This correctly guards against the attempted creation of a file with an empty name.


```java
import java.io.File;

class ContainerSizeCmpZero
{
    private static File MakeFile(String filename) {
    if(filename != null && !filename.isEmpty()) {
        return new File(filename);
    }
    return new File("default.name");
    }
}

```

## References
* Java API Specification: [ Collection.isEmpty()](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/java/util/Collection.html#isEmpty()), [ Map.isEmpty()](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/java/util/Map.html#isEmpty()), [ String.isEmpty()](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/java/lang/String.html#isEmpty()).
