# Constant array overflow
The program performs an out-of-bounds read or write operation. In addition to causing program instability, techniques exist which may allow an attacker to use this vulnerability to execute arbitrary code.


## Recommendation
Ensure that pointer dereferences are properly guarded to ensure that they cannot be used to read or write past the end of the allocation.


## Example
The first example uses a for loop which is improperly bounded by a non-strict less-than operation and will write one position past the end of the array. The second example bounds the for loop properly with a strict less-than operation.


```cpp
#define MAX_SIZE 1024

struct FixedArray {
  int buf[MAX_SIZE];
};

int main(){
  FixedArray arr;

  for(int i = 0; i <= MAX_SIZE; i++) {
    arr.buf[i] = 0; // BAD
  }

  for(int i = 0; i < MAX_SIZE; i++) {
    arr.buf[i] = 0; // GOOD
  }
}
```

## References
* CERT C Coding Standard: [ARR30-C. Do not form or use out-of-bounds pointers or array subscripts](https://wiki.sei.cmu.edu/confluence/display/c/ARR30-C.+Do+not+form+or+use+out-of-bounds+pointers+or+array+subscripts).
* OWASP: [Buffer Overflow](https://owasp.org/www-community/vulnerabilities/Buffer_Overflow).
