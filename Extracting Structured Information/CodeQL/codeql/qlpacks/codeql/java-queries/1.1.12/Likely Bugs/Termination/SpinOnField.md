# Spin on field
Repeatedly reading a non-volatile field within the condition of an empty loop statement may result in an infinite loop, since a compiler optimization may move this field access out of the loop.


## Example
In the following example, the method `spin` repeatedly tests the field `done` in a loop. The method repeats the while-loop until the value of the field `done` is set by another thread. However, the compiler could optimize the code as shown in the second code snippet, because the field `done` is not marked as `volatile` and there are no statements in the body of the loop that could change the value of `done`. The optimized version of `spin` loops forever, even when another thread would set `done` to `true`.


```java
class Spin {
    public boolean done = false;

    public void spin() {
        while(!done){
        }
    }
}

class Spin { // optimized
    public boolean done = false;

    public void spin() {
        boolean cond = done;
        while(!cond){
        }
    }
}

```

## Recommendation
Ensure that access to this field is properly synchronized. Alternatively, avoid spinning on the field and instead use the `wait` and `notifyAll` methods or the `java.util.concurrent` library to communicate between threads.


## References
* Java Language Specification: [Threads and Locks](https://docs.oracle.com/javase/specs/jls/se11/html/jls-17.html).
