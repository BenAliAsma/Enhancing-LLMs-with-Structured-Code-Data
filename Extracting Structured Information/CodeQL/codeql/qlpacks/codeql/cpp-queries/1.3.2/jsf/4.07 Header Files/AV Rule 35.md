# Missing header guard
Some header files, such as those which define structures or classes, cannot be included more than once within a translation unit, as doing so would cause a redefinition error. Such headers must be guarded to prevent ill-effects from multiple inclusion. Similarly, if header files include other header files, and this inclusion graph contains a cycle, then at least one file within the cycle must contain header guards in order to break the cycle. Because of cases like these, all headers should be guarded as a matter of good practice, even if they do not strictly need to be.

Furthermore, most modern compilers contain optimizations which are triggered by header guards. If the header guard strictly conforms to the pattern that compilers expect, then inclusions of that header other than the first have absolutely no effect: the file isn't re-read from disk, nor is it re-tokenised or re-preprocessed. This can result in a noticeable, albeit minor, improvement to compilation time.


## Recommendation
Add one of the following forms of header guard to the file (where `HEADER_NAME` is a unique identifier derived from the name of the file):

1. `#ifndef HEADER_NAME` followed by `#define HEADER_NAME` at the very start of the header, and a matching `#endif` at the very end.
1. `#if !defined(HEADER_NAME)` followed by `#define HEADER_NAME` at the very start of the header, and a matching `#endif` at the very end.
1. `#pragma once` anywhere within the header.
Note that if you are updating code to match the Joint Strike Fighter Air Vehicle coding standard, then the first option is the only appropriate form.


## Example
The author of the following header tried to use header guards, but made a typo:


```cpp
#ifndef MY_HAEDER_H
#define MY_HEADER_H

/* ... contents of my_header.h ... */

#endif
```
In scenarios like this, `MY_HAEDER_H` should be replaced by `MY_HEADER_H` (note the transposed `A` and `E`):


```cpp
#ifndef MY_HEADER_H
#define MY_HEADER_H

/* ... contents of my_header.h ... */

#endif
```

## Example
The following header would seem to be guarded, but doesn't strictly abide to the rules:


```cpp
#ifdef __linux__
#ifndef MY_LINUX_ONLY_HEADER_H
#define MY_LINUX_ONLY_HEADER_H

/* ... contents of my_linux_only_header.h ... */

#endif // MY_LINUX_ONLY_HEADER_H
#endif // __linux__
```
Although the preprocessor directives in the preceding header will prevent errors from repeated inclusion, not all compilers are intelligent enough to recognise it as being guarded. Consequently compiler optimization will be limited. To ensure that the guard is recognized by compilers, change the header so that the guard is the outermost directive:


```cpp
#ifndef MY_LINUX_ONLY_HEADER_H
#define MY_LINUX_ONLY_HEADER_H
#ifdef __linux__

/* ... contents of my_linux_only_header.h ... */

#endif // __linux__
#endif // MY_LINUX_ONLY_HEADER_H
```

## Example
The following header evolved over time, with different authors adding function declarations in different places:


```cpp
void alpha();

#ifndef MY_FUNCTIONS_H

void beta();

#define MY_FUNCTIONS_H

void gamma();
void delta();
void epsilon();

#endif

void omega();
```
Unfortunately, the result is that some declarations are before the initial `#ifndef`, some are between the `#ifndef` and the `#define`, and some are after the final `#endif`. All three of these things must be addressed to turn the file into a correctly guarded header:


```cpp
#ifndef MY_FUNCTIONS_H
#define MY_FUNCTIONS_H

void alpha();
void beta();
void gamma();
void delta();
void epsilon();
void omega();

#endif
```

## References
* AV Rules 27 and 35, *Joint Strike Fighter Air Vehicle C++ Coding Standards*. Lockheed Martin Corporation, 2005.
* [PRE06-C. Enclose header files in an inclusion guard](https://wiki.sei.cmu.edu/confluence/display/c/PRE06-C.+Enclose+header+files+in+an+include+guard)
* [Header files](http://www.learncpp.com/cpp-tutorial/19-header-files/)
* [Headers and Includes: Why and How](http://www.cplusplus.com/forum/articles/10627/)
* [The Multiple-Include Optimization](https://gcc.gnu.org/onlinedocs/cppinternals/Guard-Macros.html)
