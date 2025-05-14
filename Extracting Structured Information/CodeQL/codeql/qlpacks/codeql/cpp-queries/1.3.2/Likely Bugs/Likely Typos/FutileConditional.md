# Futile conditional
This rule finds If-statements with an empty then-branch and no else-branch. These statements are usually unimplemented skeleton code that should be implemented, or real unused code that should be removed to avoid confusion and misuse.


## Recommendation
There might be missing statements in the then-branch or the expression in the condition can be rewritten without using an If-statement.


## Example

```cpp
//empty then and else branches, will not do anything (other than the check)
if (check(i)) {
}

```
