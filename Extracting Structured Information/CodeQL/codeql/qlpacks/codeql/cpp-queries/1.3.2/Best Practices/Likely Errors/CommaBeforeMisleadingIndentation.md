# Comma before misleading indentation
If the expression after the comma operator starts at an earlier column than the expression before the comma, then this suspicious indentation possibly indicates a logic error, caused by a typo that may escape visual inspection.

> WARNING: This query has medium precision because CodeQL currently does not distinguish between tabs and spaces in whitespace. If a file contains mixed tabs and spaces, alerts may highlight code that is correctly indented for one value of tab size but not for other tab sizes.

## Recommendation
To ensure that your code is easy to read and review, use standard indentation around the comma operator. Always begin the right-hand-side operand at the same level of indentation (column number) as the left-hand-side operand. This makes it easier for other developers to see the intended behavior of your code.

Use whitespace consistently to communicate your coding intentions. Where possible, avoid mixing tabs and spaces within a file. If you need to mix them, use them consistently.


## Example
This example shows three different ways of writing the same code. The first example contains a comma instead of a semicolon which means that the final line is part of the `if` statement, even though the indentation suggests that it is intended to be separate. The second example looks different but is functionally the same as the first example. It is more likely that the developer intended to write the third example.


```cpp
/*
 * In this example, the developer intended to use a semicolon but accidentally used a comma:
 */

enum privileges entitlements = NONE;

if (is_admin)
    entitlements = FULL, // BAD

restrict_privileges(entitlements);

/*
 * The use of a comma means that the first example is equivalent to this second example:
 */

enum privileges entitlements = NONE;

if (is_admin) {
    entitlements = FULL;
    restrict_privileges(entitlements);
}

/*
 * The indentation of the first example suggests that the developer probably intended the following code:
 */

enum privileges entitlements = NONE;

if (is_admin)
    entitlements = FULL; // GOOD

restrict_privileges(entitlements);

```

## References
* Wikipedia: [Comma operator](https://en.wikipedia.org/wiki/Comma_operator)
* Wikipedia: [Indentation style &mdash; Tabs, spaces, and size of indentations](https://en.wikipedia.org/wiki/Indentation_style#Tabs,_spaces,_and_size_of_indentations)
* Common Weakness Enumeration: [CWE-1078](https://cwe.mitre.org/data/definitions/1078.html).
* Common Weakness Enumeration: [CWE-670](https://cwe.mitre.org/data/definitions/670.html).
