# Error-prone name of loop variable
This rule finds loops with an iteration variable that has a short name. The iteration variable of a nested loop should have a descriptive name: avoid short names like i, j, or k that can cause confusion except in very simple loops.


## Recommendation
Change the name of the inner loop's iteration variables to something more descriptive, to make it easier to understand code in complex, nested loops.


## Example

```cpp
int i = 0;
for (i = 0; i < NUM_RECORDS; i++) {
	int j = 0;
	//This loop should have a more descriptive iteration variable
	for (j = 0; j < NUM_FIELDS; j++) {
		process(record[i]->field[j]);
	}
	
	int field_idx = 0;
	//Better: the inner loop has a descriptive name
	for (field_idx = 0; field_idx < NUM_FIELDS; field_idx++) {
		save(record[i]->field[field_idx]);
	}
}
```

## References
* [ROS C++ Style Guide: Variables](http://wiki.ros.org/CppStyleGuide#Variables)
