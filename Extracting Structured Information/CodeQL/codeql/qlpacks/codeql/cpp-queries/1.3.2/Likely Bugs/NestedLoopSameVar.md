# Nested loops with same variable
This rule finds nested loops in which the iteration variable is the same for both loops. The behavior of the loop will be difficult to understand as the inner loop will affect the iteration variable of the outer loop; this is most likely a typo.

The rule flags the condition expression in the inner loop that uses the same variable as the iteration variable of the outer loop.


## Recommendation
If the inner loop is starting by initializing the variable to 0, it's most likely a typo and then the inner loop variable should be changed. It is good practice to use descriptive names in nested loops rather than i and j to avoid confusion. If the inner loop is merely consuming the rest of the iteration as a special case, then it's better to replace the inner for loop with a while loop and also document it.


## Example

```cpp
int x1 = 0;
for (x1 = 0; x1 < 100; x1++) {
    int x2 = 0;
    for (x1 = 0; x1 < 300; x1++) {
    // this is most likely a typo
    // the outer loop will exit immediately
    } 
}

for (x1 = 0; x1 < 100; x1++) {
  if(x1 == 10 && condition) {
    for (; x1 < 75; x1++) {
      // this should be written as a while loop
    }   
  }
}

```

## References
* Tutorialspoint - The C++ Programming Language: [C++ nested loops](http://www.tutorialspoint.com/cplusplus/cpp_nested_loops.htm)
