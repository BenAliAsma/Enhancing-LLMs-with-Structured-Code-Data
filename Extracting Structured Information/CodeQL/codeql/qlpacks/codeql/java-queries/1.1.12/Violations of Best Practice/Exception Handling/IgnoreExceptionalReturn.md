# Ignored error status of call
Many methods in the Java Development Kit (for examples, see the references below) return status values (for example, as an `int`) to indicate whether the method execution finished normally. They may return an error code if the method did not finish normally. If the method result is not checked, exceptional method executions may cause subsequent code to fail.


## Recommendation
You should insert additional code to check the return value and take appropriate action.


## Example
The following example uses the `java.io.InputStream.read` method to read 16 bytes from an input stream and store them in an array. However, `read` may not actually be able to read as many bytes as requested, for example because the stream is exhausted. Therefore, the code should not simply rely on the array `b` being filled with precisely 16 bytes from the input stream. Instead, the code should check the method's return value, which indicates the number of bytes actually read.


```java
java.io.InputStream is = (...);
byte[] b = new byte[16];
is.read(b);
```

## References
* SEI CERT Oracle Coding Standard for Java: [ EXP00-J. Do not ignore values returned by methods](https://wiki.sei.cmu.edu/confluence/display/java/EXP00-J.+Do+not+ignore+values+returned+by+methods).
* Java API Specification: [ java.util.Queue.offer](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/java/util/Queue.html#offer(E)).
* Java API Specification: [ java.util.concurrent.BlockingQueue.offer](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/java/util/concurrent/BlockingQueue.html#offer(E,long,java.util.concurrent.TimeUnit)).
* Java API Specification, java.util.concurrent.locks.Condition: [ await](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/java/util/concurrent/locks/Condition.html#await(long,java.util.concurrent.TimeUnit)), [ awaitUntil](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/java/util/concurrent/locks/Condition.html#awaitUntil(java.util.Date)), [ awaitNanos](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/java/util/concurrent/locks/Condition.html#awaitNanos(long)).
* Java API Specification, java.io.File: [ createNewFile](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/java/io/File.html#createNewFile()), [ delete](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/java/io/File.html#delete()), [ mkdir](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/java/io/File.html#mkdir()), [ renameTo](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/java/io/File.html#renameTo(java.io.File)), [ setLastModified](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/java/io/File.html#setLastModified(long)), [ setReadOnly](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/java/io/File.html#setReadOnly()), [ setWritable(boolean)](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/java/io/File.html#setWritable(boolean)), [ setWritable(boolean, boolean)](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/java/io/File.html#setWritable(boolean,boolean)).
* Java API Specification, java.io.InputStream: [ skip](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/java/io/InputStream.html#skip(long)), [ read(byte\[\])](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/java/io/InputStream.html#read(byte%5B%5D)), [ read(byte\[\], int, int)](https://docs.oracle.com/en/java/javase/11/docs/api/java.base/java/io/InputStream.html#read(byte[],int,int)).
* Common Weakness Enumeration: [CWE-391](https://cwe.mitre.org/data/definitions/391.html).
