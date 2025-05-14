# Boolean value in bitwise operation
This rule finds Boolean values (i.e. expressions that will always have the value 0 or 1) which are used in a bit-wise context.

Due to the restricted range of Boolean values, only the lowest bit could possibly be set, and thus applying a bit operation is rarely what is intended. Violations are often indicative of a typo or a misunderstanding of operator precedence. Another common source of defects is using the bitwise AND operator instead of logical AND.


## Recommendation
Carefully inspect the flagged expressions to make sure they exhibit the intended behavior. Consider adding parentheses if necessary, or replacing bitwise operations with logical equivalents. If the code is indeed correct, consider extracting the Boolean value to a variable to make it clear what is happening.


## Example

```cpp
if (!flags & SOME_BIT) { //wrong: '!' has higher precedence than '&', so this
                         // is bracketed as '(!flags) & SOME_BIT', and does not
                         // check whether a particular bit is set.
    // ...
}

if ((p != NULL) & p->f()) { //wrong: The use of '&' rather than '&&' will still
                            // de-reference the pointer even if it is NULL.
    // ...
}

int bits = (s > 8) & 0xff; //wrong: Invalid attempt to get the 8 most significant
                           // bits of a short.

```
