# Boxed variable is never null
In Java all of the primitive types have boxed counterparts. The boxed types are objects and can therefore be `null`, whereas the primitive types can never be `null`. The names of the primitive and boxed types are similar except that primitive types start with a lower-case letter and boxed types start with an upper-case letter (also, for `char` and `int` the names of the boxed types are slightly longer, namely `Character` and `Integer`).

Because the names are so similar and because Java performs automatic boxing and unboxing conversions, they can easily be confused. Furthermore, using a boxed type where a primitive type was intended leads to both readability issues and potentially superfluous allocation of objects.


## Recommendation
If a variable is never assigned `null` it should use the primitive type, as this both directly shows the impossibility of `null` and also avoids unnecessary boxing and unboxing conversions.


## Example
In the example below the variable `done` controls the loop exit. It is only set to `false` before the loop entry and set to `true` at some point during the loop iteration.


```java
Boolean done = false;
while (!done) {
  // ...
  done = true;
  // ...
}

```
Each of the assignments to `done` involves a boxing conversion and the check involves an unboxing conversion. Since `done` is never `null`, these conversions can be completely avoided, and the code made clearer, by using the primitive type instead. Therefore the code should be rewritten in the following way:


```java
boolean done = false;
while (!done) {
  // ...
  done = true;
  // ...
}

```

## References
* Java Language Specification: [Boxing Conversion](https://docs.oracle.com/javase/specs/jls/se11/html/jls-5.html#jls-5.1.7), [Unboxing Conversion](https://docs.oracle.com/javase/specs/jls/se11/html/jls-5.html#jls-5.1.8).
* The Java Tutorials: [Autoboxing and Unboxing](https://docs.oracle.com/javase/tutorial/java/data/autoboxing.html).
