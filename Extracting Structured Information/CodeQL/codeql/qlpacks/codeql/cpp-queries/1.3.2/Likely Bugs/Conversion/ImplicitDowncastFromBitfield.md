# Implicit downcast from bitfield
A bitfield may be unintentionally truncated when implicitly cast to an integer type storing fewer bits. This can lead to inaccurate iteration or allocation when the bitfield is used to count elements of a data structure, or to loss of information stored in the upper portion of the bitfield.


## Recommendation
Use the bitfield with a wider integer type, or use an explicit cast if the truncation is intended.


## Example
In the following example, a bitfield is accessed both through a method that truncates it and through direct field access. This results in a buffer overflow in the for loop.


```c
typedef struct {
	unsigned int x : 24;
} my_struct;

unsigned short getX(my_struct s ) {
	return s.x; //BAD: implicit truncation
}

unsigned int getXGood(my_struct s) {
	return s.x //GOOD: no truncation
}

int main (int argc, char **argv) {
	my_struct s;
	s.x = USHORT_MAX + 1;
	int* array = calloc(sizeof(int), getX(s)); //BAD: buffer allocated is smaller than intended
	for (int i = 0; i < s.x; i++) {
		array[i] = i;
	}

	int* array2 = calloc(sizeof(int), getXGood(s)); //GOOD
	for (int i = 0; i < s.x; i++) {
		array[i] = i;
	}
}

```

## References
* [cpp-reference.com: Bit field](http://en.cppreference.com/w/cpp/language/bit_field)
* [cpp-reference.com: Implicit conversion](http://en.cppreference.com/w/cpp/language/implicit_conversion)
* [https://cwe.mitre.org/data/definitions/681.html](https://cwe.mitre.org/data/definitions/681.html)
