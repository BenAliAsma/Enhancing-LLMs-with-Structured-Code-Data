# Lossy function result cast
This rule finds function calls whose result type is a floating point type, which are implicitly cast to an integral type. Such code may not behave as intended when the floating point return value has a fractional part, or takes an extreme value outside the range that can be represented by the integer type.


## Recommendation
Consider changing the surrounding expression to match the floating point type. If rounding is intended, explicitly round using a standard function such as `trunc`, `floor` or `round`.


## Example

```cpp
double getWidth();

void f() {
	int width = getWidth();
	
	// ...
}

```
In this example, the result of the call to `getWidth()` is implicitly cast to `int`, resulting in an unintended loss of accuracy. To fix this, the type of variable `width` could be changed from `int` to `double`.


## References
* Microsoft Visual C++ Documentation: [Type Conversions and Type Safety (Modern C++)](https://docs.microsoft.com/en-us/cpp/cpp/type-conversions-and-type-safety-modern-cpp?view=vs-2017).
* Cplusplus.com: [Type conversions](http://www.cplusplus.com/doc/tutorial/typecasting/).
