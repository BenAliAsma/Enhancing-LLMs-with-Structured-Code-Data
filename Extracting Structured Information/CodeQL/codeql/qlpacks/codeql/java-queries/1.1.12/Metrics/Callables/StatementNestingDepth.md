# Statement nesting depth
A method that contains a high level of nesting can be very difficult to understand. As noted in \[McConnell\], the human brain cannot easily handle more than three levels of nested `if` statements.


## Recommendation
Extract nested statements into new methods, for example by using the 'Extract Method' refactoring from \[Fowler\].

For more ways to reduce the level of nesting in a method, see \[McConnell\].

Furthermore, a method that has a high level of nesting often indicates that its design can be improved in other ways, as well as dealing with the nesting problem itself.


## Example
In the following example, the code has four levels of nesting and is unnecessarily difficult to read.


```java
public static void printCharacterCodes_Bad(String[] strings) {
    if (strings != null) {
        for (String s : strings) {
            if (s != null) {
                for (int i = 0; i < s.length(); i++) {
                    System.out.println(s.charAt(i) + "=" + (int) s.charAt(i));
                }
            }
        }
    }
}
```
In the following modified example, some of the nested statements have been extracted into a new method `PrintAllCharInts`.


```java
public static void printAllCharInts(String s) {
    if (s != null) {
        for (int i = 0; i < s.length(); i++) {
            System.out.println(s.charAt(i) + "=" + (int) s.charAt(i));
        }
    }
}
public static void printCharacterCodes_Good(String[] strings) {
    if (strings != null) {
        for(String s : strings){
            printAllCharInts(s);
        }
    }
}
```

## References
* M. Fowler, *Refactoring*, pp. 89-95. Addison-Wesley, 1999.
* S. McConnell, *Code Complete*, 2nd Edition, &sect;19.4. Microsoft Press, 2004.
