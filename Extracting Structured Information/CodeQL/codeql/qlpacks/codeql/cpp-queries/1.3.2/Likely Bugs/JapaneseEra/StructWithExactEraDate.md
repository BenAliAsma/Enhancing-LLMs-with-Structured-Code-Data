# Hard-coded Japanese era start date in struct
When eras change, date and time conversions that rely on a hard-coded era start date need to be reviewed. Conversions relying on Japanese dates in the current era can produce an ambiguous date. The values for the current Japanese era dates should be read from a source that will be updated, such as the Windows registry.


## References
* [The Japanese Calendar's Y2K Moment](https://blogs.msdn.microsoft.com/shawnste/2018/04/12/the-japanese-calendars-y2k-moment/).
