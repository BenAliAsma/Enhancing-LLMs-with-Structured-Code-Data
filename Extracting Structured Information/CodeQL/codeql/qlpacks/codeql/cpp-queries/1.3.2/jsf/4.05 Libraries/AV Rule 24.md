# Dangerous system functions
This query is part of a suite that tests code against the *Joint Strike Fighter Air Vehicle C++ Coding Standard* (JSF). Alerts reported by this query highlight code that may break the JSF rule listed in the References section.

The JSF rule this query tests is likely to be too strict for projects that do not follow the JSF standard.

This query highlights calls to the standard library functions `abort, exit, getenv` and `system`. The functions `abort` and `exit` should not be called as they immediately terminate the program and will bypass all the normal error and exception handling routines in the software. This is especially important in software which is run on systems without an interactive OS, as restarting the software may require a complete reboot of the system. `getenv` and `system` will usually not work at all on systems that do not have a command processor.


## Recommendation
Do not use any of the functions mentioned above. Use the error/exception handling mechanism of the software system you are using instead of `exit` or `abort`, or write your own functions to emulate the functionality provided by running OS commands through `system` and `getenv`.


## Example

```cpp
class LibraryClass {
public:
	void f() {
		...
		if (error) {
			abort(); //immediately terminates program, especially
			//bad since the class is in a library and not in the main
			//logic of the application.
		}
	}

	char* diff(string file1, string file2) {
		string command;
		command = "diff " + file1 + " " + file2;
		system(command); //call to system, may not be supported in all platforms
		...
		return cmd_out;
	}
};

```

## References
* AV Rule 24, *Joint Strike Fighter Air Vehicle C++ Coding Standards*. Lockheed Martin Corporation, 2005.
