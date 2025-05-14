# Functions with too many parameters
This rule finds functions with many parameters. Passing too many parameters to a function is a sign that the function is not cohesive (i.e. lacks a single purpose). These functions could be split into smaller, more cohesive functions.


## Recommendation
These functions could probably be refactored by wrapping related parameters into `struct`s.


## Example

```cpp
// this example has 15 parameters.
void fillRect(int x, int y, int w, int h,
              int r1, int g1, int b1, int a1,
              int r2, int g2, int b2, int a2,
              gradient_type grad, unsigned int flags, bool border)
{
    // ...
}

```

## References
* S. McConnell. *Code Complete, 2d ed*. Microsoft Press, 2004.
