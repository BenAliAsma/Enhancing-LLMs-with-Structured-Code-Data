# Rethrowing exception variable
Rethrowing an exception variable will lose the stack trace in the original exception, and replace it with the stack trace from the `throw` statement. This will make debugging the root cause of the exception more difficult, for example if the stack trace is written to a log file.


## Recommendation
Consider using `throw;` to rethrow the original exception. Not only is this simpler, but it will retain the original stack information.


## Example
This example shows an exception handler which sets the status to `UnexpectedException` if an exception is thrown. However it throws `ex` which discards the original stack trace containing the source of the error.


```csharp
try
{
  Run();
status = Status.Success;
}
catch (Exception ex)
{
  status = Status.UnexpectedException;
  throw ex;    // BAD
}

```
The fix is to replace the `throw` statement as follows:


```csharp
try
{
  Run();
status = Status.Success;
}
catch
{
  status = Status.UnexpectedException;
  throw;    // GOOD
}

```
Since the variable `ex` is no longer needed, the catch variable has been removed.


## References
* MSDN: [Creating and Throwing Exceptions (C\# Programming Guide)](https://msdn.microsoft.com/en-us/library/ms173163.aspx).
* MSDN: [throw (C\# Reference)](https://msdn.microsoft.com/en-us/library/1ah5wsex.aspx).
