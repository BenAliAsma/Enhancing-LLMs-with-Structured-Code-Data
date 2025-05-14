# Year field changed using an arithmetic operation without checking for leap year
The leap year rule for the Gregorian calendar, which has become the internationally accepted civil calendar, is: every year that is exactly divisible by four is a leap year, except for years that are exactly divisible by 100, but these centurial years are leap years if they are exactly divisible by 400.

A leap year bug occurs when software (in any language) is written without consideration of leap year logic, or with flawed logic to calculate leap years; which typically results in incorrect results.

The impact of these bugs may range from almost unnoticeable bugs such as an incorrect date, to severe bugs that affect reliability, availability or even the security of the affected system.

When performing arithmetic operations on a variable that represents a year, it is important to consider that the resulting value may not be a valid date.

The typical example is doing simple year arithmetic (i.e. `date.year++`) without considering if the resulting value will be a valid date or not.


## Recommendation
When modifying a year field on a date structure, verify if the resulting year is a leap year.


## Example
In this example, we are adding 1 year to the current date. This may work most of the time, but on any given February 29th, the resulting value will be invalid.


```c
SYSTEMTIME st;
FILETIME ft;
GetSystemTime(&st);

// Flawed logic may result in invalid date
st.wYear++;

// The following code may fail
SystemTimeToFileTime(&st, &ft);
```
To fix this bug, check the result for leap year.


```c
SYSTEMTIME st;
FILETIME ft;
GetSystemTime(&st);

// Flawed logic may result in invalid date
st.wYear++;

// Check for leap year, and adjust the date accordingly
bool isLeapYear = st.wYear % 4 == 0 && (st.wYear % 100 != 0 || st.wYear % 400 == 0);
st.wDay = st.wMonth == 2 && st.wDay == 29 && !isLeapYear ? 28 : st.wDay;

if (!SystemTimeToFileTime(&st, &ft))
{
	// handle error
}

```

## References
* NASA / Goddard Space Flight Center - [Calendars](https://eclipse.gsfc.nasa.gov/SEhelp/calendars.html)
* Wikipedia - [ Leap year bug](https://en.wikipedia.org/wiki/Leap_year_bug)
* Microsoft Azure blog - [ Is your code ready for the leap year?](https://azure.microsoft.com/en-us/blog/is-your-code-ready-for-the-leap-year/)
