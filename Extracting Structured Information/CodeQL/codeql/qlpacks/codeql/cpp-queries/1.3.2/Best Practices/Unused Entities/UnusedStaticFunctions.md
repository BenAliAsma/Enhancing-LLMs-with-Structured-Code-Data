# Unused static function
This rule finds static functions with definitions that are never called or accessed. These unused functions should be removed to increase code readability, reduce object size and avoid misuse.


## Recommendation
Removing these unused static functions will make code more readable. A common pitfall is that code using a static function is guarded by conditional compilation but the static function is not. Notice that this detects directly unused functions and removing a static function may expose more unused functions.


## Example

```cpp
//start of file
static void f() { //static function f() is unused in the file
    //...
}
static void g() {
    //...
}
void public_func() { //non-static function public_func is not called in file, 
                     //but could be visible in other files
    //...
    g(); //call to g()
    //...
}
//end of file

```

## References
* [MSC12-C. Detect and remove code that has no effect](https://wiki.sei.cmu.edu/confluence/display/c/MSC12-C.+Detect+and+remove+code+that+has+no+effect+or+is+never+executed)
* Common Weakness Enumeration: [CWE-561](https://cwe.mitre.org/data/definitions/561.html).
