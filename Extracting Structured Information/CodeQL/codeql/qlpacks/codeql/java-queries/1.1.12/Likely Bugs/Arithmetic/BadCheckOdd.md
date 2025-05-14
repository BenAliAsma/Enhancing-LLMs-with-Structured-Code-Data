# Bad parity check
Avoid using `x % 2 == 1` or `x % 2 > 0` to check whether a number `x` is odd, or `x % 2 != 1` to check whether it is even. Such code does not work for negative numbers. For example, `-5 % 2` equals `-1`, not `1`.


## Recommendation
Consider using `x % 2 != 0` to check for odd and `x % 2 == 0` to check for even.


## Example
-9 is an odd number but this example does not detect it as one. This is because `-9 % 2 ` is -1, not 1.


```java
class CheckOdd {
    private static boolean isOdd(int x) {
        return x % 2 == 1;
    }
    
    public static void main(String[] args) {
        System.out.println(isOdd(-9)); // prints false
    }
}
```
It would be better to check if the number is even and then invert that check.


```java
class CheckOdd {
    private static boolean isOdd(int x) {
        return x % 2 != 0;
    }
    
    public static void main(String[] args) {
        System.out.println(isOdd(-9)); // prints true
    }
}
```

## References
* J. Bloch and N. Gafter, *Java Puzzlers: Traps, Pitfalls, and Corner Cases*, Puzzle 1. Addison-Wesley, 2005.
* Java Language Specification: [Remainder Operator %](https://docs.oracle.com/javase/specs/jls/se11/html/jls-15.html#jls-15.17.3).
