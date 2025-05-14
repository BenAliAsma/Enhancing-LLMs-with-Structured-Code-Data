# Use of goto
Code should be restricted to very simple control flow constructs; in particular, the `goto` statement should not be used. Simpler control flow translates into stronger capabilities for analysis and often results in improved code clarity.


## Recommendation
Usually it is clearer to re-write uses of the `goto` statement in terms of the more structured control-flow constructs like `if`/`else`, `switch` statements or loops in conjunction with `break` or `continue`.


## Example

```c
int matching; int i;
for (i = 0; i < numItems; i++) {
	if (items[i] == item) {
		matching = i;
		goto found; // Problematic 'goto': Use 'break' instead.
	}
}
found:
// ... use 'matching' ...

```
