# Non-terminated if-else-if chain
An `if-else-if` statement without a terminating `else` clause may allow execution to 'fall through' silently, if none of the `if` clauses are matched.


## Recommendation
Include a terminating `else` clause to `if-else-if` statements to prevent execution from falling through silently. If the terminating `else` clause is intended to be unreachable code, it is advisable that it throws a `RuntimeException` to alert the user of an internal error.


## Example
In the following example, the `if` statement outputs the grade that is achieved depending on the test score. However, if the score is less than 60, execution falls through silently.


```java
int score;
char grade;

// ...

if (score >= 90) {
    grade = 'A';
} else if (score >= 80) {
    grade = 'B';
} else if (score >= 70) {
    grade = 'C';
} else if (score >= 60) {
    grade = 'D';
  // BAD: No terminating 'else' clause
}
System.out.println("Grade = " + grade);
```
In the following modified example, the `if` statement includes a terminating `else` clause, to allow for scores that are less than 60.


```java
int score;
char grade;

// ...

if (score >= 90) {
    grade = 'A';
} else if (score >= 80) {
    grade = 'B';
} else if (score >= 70) {
    grade = 'C';
} else if (score >= 60) {
    grade = 'D';
} else {  // GOOD: Terminating 'else' clause for all other scores
    grade = 'F';
}
System.out.println("Grade = " + grade);
```

## References
* Java SE Documentation: [7.4 if, if-else, if else-if else Statements](https://www.oracle.com/java/technologies/javase/codeconventions-statements.html#449).
