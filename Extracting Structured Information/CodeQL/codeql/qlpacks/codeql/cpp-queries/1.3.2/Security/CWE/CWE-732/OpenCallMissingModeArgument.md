# File opened with O_CREAT flag but without mode argument
When opening a file with the `O_CREAT` or `O_TMPFILE` flag, the `mode` must be supplied. If the `mode` argument is omitted, some arbitrary bytes from the stack will be used as the file mode. This leaks some bits from the stack into the permissions of the file.


## Recommendation
The `mode` must be supplied when `O_CREAT` or `O_TMPFILE` is specified.


## Example
The first example opens a file with the `O_CREAT` flag without supplying the `mode` argument. In this case arbitrary bytes from the stack will be used as `mode` argument. The second example correctly supplies the `mode` argument and creates a file that is user readable and writable.


```c
int open_file_bad() {
	// BAD - this uses arbitrary bytes from the stack as mode argument
        return open(FILE, O_CREAT)
}

int open_file_good() {
	// GOOD - the mode argument is supplied
        return open(FILE, O_CREAT, S_IRUSR | S_IWUSR)
}

```
