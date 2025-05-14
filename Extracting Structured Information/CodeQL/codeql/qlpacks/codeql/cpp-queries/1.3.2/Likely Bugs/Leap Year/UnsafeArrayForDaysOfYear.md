# Unsafe array for days of the year
The leap year rule for the Gregorian calendar, which has become the internationally accepted civil calendar, is: every year that is exactly divisible by four is a leap year, except for years that are exactly divisible by 100, but these centurial years are leap years if they are exactly divisible by 400.

A leap year bug occurs when software (in any language) is written without consideration of leap year logic, or with flawed logic to calculate leap years; which typically results in incorrect results.

The impact of these bugs may range from almost unnoticeable bugs such as an incorrect date, to severe bugs that affect reliability, availability or even the security of the affected system.

This query helps to detect when a developer allocates an array or other fixed-length data structure such as `std::vector` with 365 elements – one for each day of the year.

Since leap years have 366 days, there will be no allocated element on December 31st at the end of a leap year; which will lead to a buffer overflow on a leap year.


## Recommendation
When allocating memory for storing elements for each day of the year, ensure that the correct number of elements are allocated.

It is also highly recommended to compile the code with array-bounds checking enabled whenever possible.


## Example
In this example, we are allocating 365 integers, one for each day of the year. This code will fail on a leap year, when there are 366 days.


```c
int items[365];
items[dayOfYear - 1] = x; // buffer overflow on December 31st of any leap year 
```
When using arrays, allocate the correct number of elements to match the year.


```c
bool isLeapYear = year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
int *items = new int[isLeapYear ? 366 : 365];
// ...
delete[] items;
```

## References
* NASA / Goddard Space Flight Center - [Calendars](https://eclipse.gsfc.nasa.gov/SEhelp/calendars.html)
* Wikipedia - [ Leap year bug](https://en.wikipedia.org/wiki/Leap_year_bug)
* Microsoft Azure blog - [ Is your code ready for the leap year?](https://azure.microsoft.com/en-us/blog/is-your-code-ready-for-the-leap-year/)
