# Array argument size mismatch
This rule looks for function calls where the size of the array being passed is smaller than the size of the declared array parameter for the function. This is most likely going to lead to memory accesses that are beyond the bounds of the passed array.

The size of the array being passed is very likely wrong as it does not match that of the function's parameter.


## Recommendation
Check the array being passed to the function is correct, or modify its size to match that of the function's array parameter.


## Example

```cpp

//Function foo's array parameter has a specified size
void foo(int a[10]) {
	int i = 0;
	for (i = 0; i <10; i++) {
		a[i] = i * 2;
	}
}

...

int my_arr[5];
foo(my_arr); //my_arr is smaller than foo's array parameter, and will cause access to memory outside its bounds
```
