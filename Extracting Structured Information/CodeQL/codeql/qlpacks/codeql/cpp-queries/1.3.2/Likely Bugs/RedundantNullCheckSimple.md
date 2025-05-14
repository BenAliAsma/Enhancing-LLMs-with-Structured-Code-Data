# Redundant null check due to previous dereference
This rule finds comparisons of a pointer to null that occur after a reference of that pointer. It's likely either the check is not required and can be removed, or it should be moved to before the dereference so that a null pointer dereference does not occur.


## Recommendation
The check should be moved to before the dereference, in a way that prevents a null pointer value from being dereferenced. If it's clear that the pointer cannot be null, consider removing the check instead.


## Example

```cpp
int f(MyList *list) {
	list->append(1);

	// ...

	if (list != NULL)
	{
		list->append(2);
	}
}

```

## References
* [ Null Dereference ](https://www.owasp.org/index.php/Null_Dereference)
* Common Weakness Enumeration: [CWE-476](https://cwe.mitre.org/data/definitions/476.html).
