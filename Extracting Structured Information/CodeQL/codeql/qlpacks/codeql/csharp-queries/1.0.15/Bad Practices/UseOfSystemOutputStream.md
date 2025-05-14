# Poor logging: use of system output stream
Writing directly to system output streams is often used as an unstructured form of logging. A proper logging mechanism is a better way to direct messages to the desired location and also ensures that no critical information is leaked to the standard outputs. The rule points out any call to the `Console.Write*(...)` methods and any access to `Console.Out` or `Console.Error`.


## Recommendation
Use proper logging with a dedicated API. Alternatively redirect standard output and error streams by calling `Console.SetOut(...)` and `Console.SetError(...)`.

