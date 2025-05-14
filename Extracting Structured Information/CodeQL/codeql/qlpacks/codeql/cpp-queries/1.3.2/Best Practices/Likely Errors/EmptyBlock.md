# Empty branch of conditional
This rule finds empty blocks that occur as a branch of a conditional or as a loop body. This may indicate badly maintained code or a defect due to an unhandled case. It is common to find commented-out code in the empty body. Commented-out code is discouraged and is a source of defects and maintainability issues.


## Recommendation
If the conditional or loop is useless, remove it.

If only the else-branch of an `if` statement is empty, omit it. If the then-branch is empty, invert the sense of the condition.


## Example

```cpp
void f(int i) {
	if (i == 10); //empty then block
		... //won't be part of the if statement

	if (i == 12) {
		...
	} else { //empty else block, most likely a mistake
	}
}

```
