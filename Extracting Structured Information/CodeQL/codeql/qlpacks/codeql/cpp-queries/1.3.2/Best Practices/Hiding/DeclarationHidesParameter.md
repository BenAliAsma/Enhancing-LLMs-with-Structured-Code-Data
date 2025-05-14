# Declaration hides parameter
This rule finds declarations of local variables that hide parameters of the surrounding function. Such declarations create variables with the same name but different scopes. This makes it hard to understand which variable is actually being used in an expression.


## Recommendation
Consider changing the name of either the variable or the parameter to keep them distinct.


## Example

```cpp
void f(int i) {
  for (int i = 0; i < 10; ++i) { //the loop variable hides the parameter to f()
    ...
  }
}


```

## References
* B. Stroustrup. *The C++ Programming Language Special Edition* p 82. Addison Wesley. 2000.
