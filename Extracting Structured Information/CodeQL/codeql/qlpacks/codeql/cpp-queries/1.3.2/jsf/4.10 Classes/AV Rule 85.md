# Opposite operator definition
This query is part of a suite that tests code against the *Joint Strike Fighter Air Vehicle C++ Coding Standard* (JSF). Alerts reported by this query highlight code that may break the JSF rule listed in the References section.

The JSF rule this query tests is likely to be too strict for projects that do not follow the JSF standard.

This query ensures that all operators with opposites (e.g. == and !=) are both defined, and that one of them is defined in terms of the other. This just enforces the consistency of meaning of the operators.

The indicated operator either does not have its opposite defined, or both of the definitions are not in terms of the other. Not defining an operator in terms of its opposite is prone to mistakes, as it requires modification of both operators when the behavior of one changes. Deliberately defining opposite operators with behaviors that are not actual logical opposites (e.g. defining `x == y` if `x` and `y` are divisible by 2 and ` x != y ` if `x` and `y` are divisible by 3) violates the almost universal assumptions developers have on the relationship of `==` and `!=` and will lead to unnecessary confusion.


## Recommendation
Make sure that both opposite operators are defined when they are overloaded, and ensure that one of the overloads is defined in terms of the other.


## Example

```cpp
class B {
	int x;
	int y; //new field
	bool operator==(const C& other) {
		return this->x % 3 == 0 && this->y % 4 ==0; //updated to include the new field y
	}
	bool operator!=(const C& other) {
		return this->x % 3 != 0; //Wrong: forgot to update to include new field y
	}
};
class C {
	int x;
	int y; //new field
	bool operator==(const C& other) {
		return this->x % 3 == 0 && this->y % 4 == 0; //updated to include the new field
	}
	bool operator!=(const C& other) {
		return !(*this == other); //Correct, no need to update this operator 
		                          //definition when adding the new field
	}
};

```

## References
* AV Rule 85, *Joint Strike Fighter Air Vehicle C++ Coding Standards*. Lockheed Martin Corporation, 2005.
