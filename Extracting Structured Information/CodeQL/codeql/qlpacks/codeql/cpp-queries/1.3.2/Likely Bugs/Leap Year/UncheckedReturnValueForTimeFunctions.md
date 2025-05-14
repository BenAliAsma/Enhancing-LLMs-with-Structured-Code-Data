# Unchecked return value for time conversion function
The leap year rule for the Gregorian calendar, which has become the internationally accepted civil calendar, is: every year that is exactly divisible by four is a leap year, except for years that are exactly divisible by 100, but these centurial years are leap years if they are exactly divisible by 400.

A leap year bug occurs when software (in any language) is written without consideration of leap year logic, or with flawed logic to calculate leap years; which typically results in incorrect results.

The impact of these bugs may range from almost unnoticeable bugs such as an incorrect date, to severe bugs that affect reliability, availability or even the security of the affected system.

When using a function that transforms a date structure, and the year on the input argument for the API has been manipulated, it is important to check for the return value of the function to make sure it succeeded.

Otherwise, the function may have failed, and the output parameter may contain invalid data that can cause any number of problems on the affected system.

The following is a list of the functions that this query covers:

* `FileTimeToSystemTime`
* `SystemTimeToFileTime`
* `SystemTimeToTzSpecificLocalTime`
* `SystemTimeToTzSpecificLocalTimeEx`
* `TzSpecificLocalTimeToSystemTime`
* `TzSpecificLocalTimeToSystemTimeEx`
* `RtlLocalTimeToSystemTime`
* `RtlTimeToSecondsSince1970`
* `_mkgmtime`

## Recommendation
When calling an API that transforms a date variable that was manipulated, always check for the return value to verify that the API call succeeded.


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
To fix this bug, you must verify the return value for `SystemTimeToFileTime` and handle any potential error accordingly.


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
