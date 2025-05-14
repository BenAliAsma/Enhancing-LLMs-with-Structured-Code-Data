# Duplicate include guard
A common pattern in header files is to use pre-processor directives to guard a header file against being processed more than once per translation unit. This practice is intended to prevent compilation errors. However, pre-processor include guards are prone to human error themselves because each include guard must be assigned a unique macro name to function correctly. If two header files share the same guard macro, the compiler may unexpectedly skip the second file it encounters, leading to compilation errors or configuration bugs.

The query will flag the pre-processor `#ifndef` directive at the beginning of any include guard that matches another include guard in the project. Browsing the list of results you will be able to find the other directive(s) which use the same macro.


## Recommendation
First decide whether the duplicate include guard is dangerous. A duplicate include guard may cause the header file to be skipped over when it shouldn't be, but occasionally this design is used on purpose to 'override' an existing header file.

To address the issue, rename the macros used by all but one instance of the duplicate include guard. Remember to change both the `#ifndef` and the `#define` directive to use the new macro name. Alternatively, consider using the `#pragma once` directive to prevent multiple inclusion without the need to define unique macros.


## Example
Here's an example of two header files that have accidentally been given the same include guard macro. To fix the issue, rename both occurrences of the macro in the second file, for example to ANOTHER_HEADER_FILE_H.


```cpp
// header_file.h

#ifndef HEADER_FILE_H
#define HEADER_FILE_H

	// ...

#endif // HEADER_FILE_H
```

```cpp
// another_header_file.h

#ifndef HEADER_FILE_H // should be ANOTHER_HEADER_FILE_H
#define HEADER_FILE_H // should be ANOTHER_HEADER_FILE_H

	// ...

#endif // HEADER_FILE_H
```

## References
* Wikipedia: [Include guard](http://en.wikipedia.org/wiki/Include_guard)
