# Arithmetic operation assumes 365 days per year
The leap year rule for the Gregorian calendar, which has become the internationally accepted civil calendar, is: every year that is exactly divisible by four is a leap year, except for years that are exactly divisible by 100, but these centurial years are leap years if they are exactly divisible by 400.

A leap year bug occurs when software (in any language) is written without consideration of leap year logic, or with flawed logic to calculate leap years; which typically results in incorrect results.

The impact of these bugs may range from almost unnoticeable bugs such as an incorrect date, to severe bugs that affect reliability, availability or even the security of the affected system.

When performing arithmetic operations on a variable that represents a date, leap years must be taken into account. It is not safe to assume that a year is 365 days long.


## Recommendation
Determine whether the time span in question contains a leap day, then perform the calculation using the correct number of days. Alternatively, use an established library routine that already contains correct leap year logic.


## References
* NASA / Goddard Space Flight Center - [Calendars](https://eclipse.gsfc.nasa.gov/SEhelp/calendars.html)
* Wikipedia - [ Leap year bug](https://en.wikipedia.org/wiki/Leap_year_bug)
* Microsoft Azure blog - [ Is your code ready for the leap year?](https://azure.microsoft.com/en-us/blog/is-your-code-ready-for-the-leap-year/)
