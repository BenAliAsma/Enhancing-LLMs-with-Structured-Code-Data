# Multiplication result may overflow and be used in allocation
The result of a multiplication is used in the size of an allocation. If the multiplication can be made to overflow, a much smaller amount of memory may be allocated than the rest of the code expects. This may lead to overflowing writes when the buffer is accessed later.


## Recommendation
To fix this issue, ensure that the arithmetic used in the size of an allocation cannot overflow before memory is allocated.


## Example
In the following example, an array of size `width * height` is allocated and stored as `pixels`. If `width` and `height` are set such that the multiplication overflows and wraps to a small value (say, 4) then the initialization code that follows the allocation will write beyond the end of the array.


```cpp

image::image(int width, int height)
{
	int x, y;

	// allocate width * height pixels
	pixels = new uint32_t[width * height];

	// fill width * height pixels
	for (y = 0; y < height; y++)
	{
		for (x = 0; x < width; x++)
		{
			pixels[(y * width) + height] = 0;
		}
	}

	// ...
}

```

## References
* Cplusplus.com: [Integer overflow](http://www.cplusplus.com/articles/DE18T05o/).
* Common Weakness Enumeration: [CWE-190](https://cwe.mitre.org/data/definitions/190.html).
* Common Weakness Enumeration: [CWE-128](https://cwe.mitre.org/data/definitions/128.html).
