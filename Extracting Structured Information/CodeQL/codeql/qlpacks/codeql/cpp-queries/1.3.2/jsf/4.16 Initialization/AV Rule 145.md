# Irregular enum initialization
This rule finds enumerations that initialize their members inconsistently. Only the first enumeration member should be initialized using '=', or all of them should be initialized. Inconsistent initialization of members in enumerations can easily cause defects, especially when adding or removing members, or if the code relies on the actual integer values of the enumeration members.


## Recommendation
Change the enumeration so that only the first member is initialized, or initialize all the members of the enumeration. In general, initialize all enumeration members if you actually care about their integer values. If not, it is better to let the compiler assign values at compile time.


## Example

```cpp
//wrong: inconsistent initialization, only the first should be initialized, 
//or all should be initialized
enum {
	VALUE_SHOULD_BE_10 = 10,
	VALUE_SHOULD_BE_11,
	VALUE_SHOULD_BE_12,
	VALUE_SHOULD_BE_20, //newly added member, but initialization was forgotten 
	                    //(would have a value of 13 instead of 20).
	VALUE_SHOULD_BE_30 = 30,
	VALUE_SHOULD_BE_40 = 40
} bad_values;

//correct: all enum values initialized
enum {
	VALUE_SHOULD_BE_10 = 10,
	VALUE_SHOULD_BE_11 = 11,
	VALUE_SHOULD_BE_12 = 12,
	VALUE_SHOULD_BE_20 = 20, //newly added member, it is less likely to forget
	                         //putting in initialization since everything else 
	                         //is initialized
	VALUE_SHOULD_BE_30 = 30,
	VALUE_SHOULD_BE_40 = 40
} good_values;

```

## References
* AV Rule 145, *Joint Strike Fighter Air Vehicle C++ Coding Standards*. Lockheed Martin Corporation, 2005.
* MISRA C++ Rule 8-5-3, *Guidelines for the use of the C++ language in critical systems*. The Motor Industry Software Reliability Associate, 2008.
* MSDN Library: [Enumerations (C++)](https://docs.microsoft.com/en-us/cpp/cpp/enumerations-cpp).
