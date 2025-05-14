# Missing default case in switch
A `switch` statement without a `default` case may allow execution to 'fall through' silently, if no cases are matched.


## Recommendation
In a `switch` statement that is based on a variable of a non-enumerated type, include a `default` case to prevent execution from falling through silently when no cases are matched. If the `default` case is intended to be unreachable code, it is advisable that it throws a `RuntimeException` to alert the user of an internal error.


## Example
In the following example, the `switch` statement outputs the menu choice that the user has made. However, if the user does not choose 1, 2, or 3, execution falls through silently.


```java
int menuChoice;

// ...

switch (menuChoice) {
    case 1:
        System.out.println("You chose number 1.");
        break;
    case 2:
        System.out.println("You chose number 2.");
        break;
    case 3:
        System.out.println("You chose number 3.");
        break;
    // BAD: No 'default' case
}
```
In the following modified example, the `switch` statement includes a `default` case, to allow for the user making an invalid menu choice.


```java
int menuChoice;

// ...

switch (menuChoice) {
    case 1:
        System.out.println("You chose number 1.");
        break;
    case 2:
        System.out.println("You chose number 2.");
        break;
    case 3:
        System.out.println("You chose number 3.");
        break;
    default:  // GOOD: 'default' case for invalid choices
        System.out.println("Sorry, you made an invalid choice.");
        break;
}
```

## References
* Java SE Documentation: [7.8 switch Statements](https://www.oracle.com/java/technologies/javase/codeconventions-statements.html#468).
* Common Weakness Enumeration: [CWE-478](https://cwe.mitre.org/data/definitions/478.html).
